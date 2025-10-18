# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
DCH Pipeline — modular orchestration for ingestion, DHG construction, traversal, and plasticity.

This module provides:
- Config dataclasses for DHG, traversal, and plasticity
- DCHPipeline class to run a step over a batch of events
- Convenience constructor using default in-memory backends

Key integrations
- Interfaces: dch_core.interfaces (events, hypergraph ops, DHG, traversal, plasticity)
- Defaults:
    - HypergraphOps: dch_core.hypergraph_mem.InMemoryHypergraph
    - GraphConnectivity: dch_core.hypergraph_mem.StaticGraphConnectivity
    - DHGConstructor: dch_core.dhg.DefaultDHGConstructor
    - TraversalEngine: dch_core.traversal.DefaultTraversalEngine
    - PlasticityEngine: dch_core.plasticity.DefaultPlasticityEngine
    - Encoder (optional): dch_data.encoders.SimpleBinnerEncoder with dch_snn.interface.EncoderConfig

Typical flow in one step()
1) Ingest events -> vertices
2) For each new head vertex:
   - Compute temporal window [t_head - delay_max, t_head - delay_min]
   - Generate TC-kNN candidates and admit into hypergraph
3) If target vertices provided:
   - Backward traverse for each target
   - Update plasticity based on hyperpaths (evidence), then prune

References
- Specification: docs/AlgorithmSpecs.md
- Interfaces: dch_core/interfaces.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Set
import os
from datetime import datetime

# Optional torch import for @torch.no_grad decorator
try:
    import torch
    def no_grad_decorator(func):
        return torch.no_grad()(func)
except ImportError:
    def no_grad_decorator(func):
        return func

from dch_core.interfaces import (
    Event,
    Vertex,
    VertexId,
    Window,
    HypergraphOps,
    GraphConnectivity,
    DHGConstructor,
    TraversalEngine,
    PlasticityEngine,
    PlasticityState,
    Hyperpath,
)
from dch_core.hypergraph_mem import InMemoryHypergraph, StaticGraphConnectivity
from dch_core.dhg import DefaultDHGConstructor
from dch_core.traversal import DefaultTraversalEngine
from dch_core.plasticity import DefaultPlasticityEngine
from dch_data.encoders import SimpleBinnerEncoder
from dch_snn.interface import EncoderConfig
from dch_core.fsm import StreamingFSMEngine
from dch_core.abstraction import DefaultAbstractionEngine, AbstractionParams
from dch_core.embeddings.wl import WLHyperpathEmbedding, WLParams
from dch_core.dpo import DPOEngine, DPOGraphAdapter, DPO_Rule, DPO_LKR, DPO_Match


# -------------------------
# Config dataclasses
# -------------------------


@dataclass(frozen=True)
class DHGConfig:
    # TC-kNN parameters
    k: int = 3
    combination_order_max: int = 2
    causal_coincidence_delta: int = 500  # time proximity for grouping (same units as timestamps)
    budget_per_head: int = 16
    init_reliability: float = 0.10
    refractory_rho: int = 0
    # Temporal admissibility window relative to head time
    delay_min: int = 100
    delay_max: int = 500


@dataclass(frozen=True)
class TraversalConfig:
    horizon: int = 2000
    beam_size: int = 8
    length_penalty_base: float = 0.98  # passed into DefaultTraversalEngine


@dataclass(frozen=True)
class PlasticityConfig:
    ema_alpha: float = 0.10
    reliability_min: float = 0.02
    reliability_max: float = 0.98
    decay_lambda: float = 0.0
    prune_threshold: float = 0.05
    # Implementation knob: "ema" (default, back-compat) or "beta"
    impl: str = "ema"

    def to_state(self, freeze: bool = False) -> PlasticityState:
        return PlasticityState(
            ema_alpha=self.ema_alpha,
            reliability_clamp=(self.reliability_min, self.reliability_max),
            decay_lambda=self.decay_lambda,
            freeze=freeze,
            prune_threshold=self.prune_threshold,
        )


@dataclass(frozen=True)
class FSMConfig:
    theta: float = 5.0
    lambda_decay: float = 0.0
    hold_k: int = 2
    min_weight: float = 1e-6
    # Cap promotions per step to keep tests fast and work bounded
    promotion_limit_per_step: int = 1


@dataclass(frozen=True)
class SNNConfig:
    enabled: bool = False
    model: str = "norse_lif"
    unroll: bool = True
    device: str = "cpu"
    # Optional additional model parameters forwarded into the Norse wrapper factory
    model_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManifoldConfig:
    enable: bool = False
    impl: str = "noop"
    log_calls: bool = False  # reserved for future debugging; no functional effect in P2-11


@dataclass(frozen=True)
class DualProofConfig:
    enable: bool = False
    mode: str = "soft"  # allowed: "soft" or "hard"
    check_points: Tuple[str, ...] = ("grow", "backward")  # default both

@dataclass(frozen=True)
class DPOConfig:
    enable: bool = False
    theta_prune: float = 0.2
    theta_freeze: float = 0.95
    apply_ops: Tuple[str, ...] = ("grow", "prune", "freeze")


@dataclass(frozen=True)
class PipelineConfig:
    dhg: DHGConfig = DHGConfig()
    traversal: TraversalConfig = TraversalConfig()
    plasticity: PlasticityConfig = PlasticityConfig()
    # FSM and abstraction wiring (feature-gated)
    fsm: FSMConfig = FSMConfig()
    enable_abstraction: bool = False
    # Passed into AbstractionParams(**abstraction_params)
    abstraction_params: Mapping[str, Any] = field(default_factory=dict)
    # Back-compat alias for per-step promotion cap
    fsm_promotion_limit_per_step: int = 1
    # Optional SNN integration (torch-optional, lazy model instantiation)
    snn: SNNConfig = SNNConfig()
    # Optional manifold feasibility interface (non-enforcing; default disabled)
    manifold: Optional[ManifoldConfig] = None
    # Optional dual-proof gating (manifold + causal) — default OFF
    dual_proof: Optional[DualProofConfig] = None
    # Optional episode audit trail log path (if set, enable EAT logger)
    audit_log_path: Optional[str] = None
    # Optional CRC JSONL log path (enabled when abstraction is also enabled)
    crc_log_path: Optional[str] = None
    # Optional DPO routing (default OFF)
    dpo: Optional[DPOConfig] = None


# -------------------------
# Pipeline
# -------------------------


class DCHPipeline:
    """
    Orchestrates DCH flow for a batch of events.

    Attributes
    - hypergraph: HypergraphOps backend
    - connectivity: GraphConnectivity oracle
    - dhg: DHGConstructor
    - traversal: TraversalEngine
    - plasticity: PlasticityEngine
    - cfg: PipelineConfig
    """

    def __init__(
        self,
        hypergraph: HypergraphOps,
        connectivity: GraphConnectivity,
        dhg: DHGConstructor,
        traversal: TraversalEngine,
        plasticity: PlasticityEngine,
        cfg: PipelineConfig = PipelineConfig(),
    ) -> None:
        self.hypergraph = hypergraph
        self.connectivity = connectivity
        self.dhg = dhg
        self.traversal = traversal
        self.plasticity = plasticity
        self.cfg = cfg

        # Optional engines set by from_defaults when enable_abstraction=True
        self.fsm_engine: Optional[StreamingFSMEngine] = None
        self.abstraction_engine: Optional[DefaultAbstractionEngine] = None
        self._wl_canonizer: Optional[WLHyperpathEmbedding] = None

        # Optional encoder and SNN integration (torch-optional; created lazily)
        self.encoder: Optional[SimpleBinnerEncoder] = None
        self.snn_model: Optional[Any] = None
        self._snn_input_size: Optional[int] = None
        self._snn_enabled: bool = False
        self._snn_device: str = "cpu"
        self._snn_model_config: Mapping[str, Any] = {}
        # Optional EAT audit logger (attached by from_defaults when configured)
        self._audit = None
        # Optional CRC extractor/logger (attached by from_defaults when configured)
        self._crc_extractor = None
        self._crc_logger = None
        # Optional manifold backend (wired by from_defaults when configured)
        self.manifold: Optional[Any] = None

        # Optional DPO engine/adapter (lazy)
        self._dpo_engine: Optional[DPOEngine] = None
        self._dpo_adapter: Optional[DPOGraphAdapter] = None

    # -------------
    # Dual-proof helpers (no-op safe)
    # -------------
    def _dual_proof_enabled(self, point: str) -> bool:
        """
        Return True if dual-proof gating is enabled and configured for the given checkpoint,
        and a manifold backend is present.
        """
        try:
            dp = getattr(self.cfg, "dual_proof", None)
            if dp is None or not getattr(dp, "enable", False):
                return False
            cps = getattr(dp, "check_points", ("grow", "backward"))
            cps_set = set(cps) if isinstance(cps, (list, tuple, set)) else set()
            return (str(point) in cps_set) and (self.manifold is not None)
        except Exception:
            return False

    def _dp_mode(self) -> str:
        """
        Return 'hard' or 'soft' (default soft) for gating mode.
        """
        dp = getattr(self.cfg, "dual_proof", None)
        mode = str(getattr(dp, "mode", "soft")).lower() if dp is not None else "soft"
        return "hard" if mode == "hard" else "soft"

    # -------------
    # DPO helpers (opt-in, default OFF)
    # -------------
    def _dpo_enabled(self) -> bool:
        try:
            dpo = getattr(self.cfg, "dpo", None)
            return bool(dpo is not None and getattr(dpo, "enable", False))
        except Exception:
            return False

    def _dpo_apply_ops(self) -> Set[str]:
        try:
            dpo = getattr(self.cfg, "dpo", None)
            ops = getattr(dpo, "apply_ops", ("grow", "prune", "freeze")) if dpo is not None else ()
            if isinstance(ops, (list, tuple, set)):
                return {str(op).lower() for op in ops}
            return {str(ops).lower()} if ops else set()
        except Exception:
            return set()

    def _ensure_dpo(self) -> None:
        if getattr(self, "_dpo_engine", None) is None or getattr(self, "_dpo_adapter", None) is None:
            try:
                dpo = getattr(self.cfg, "dpo", None)
                theta_prune = float(getattr(dpo, "theta_prune", 0.2)) if dpo is not None else 0.2
                theta_freeze = float(getattr(dpo, "theta_freeze", 0.95)) if dpo is not None else 0.95
            except Exception:
                theta_prune, theta_freeze = 0.2, 0.95
            self._dpo_engine = DPOEngine(theta_prune=theta_prune, theta_freeze=theta_freeze)
            try:
                self._dpo_adapter = DPOGraphAdapter(self.hypergraph)  # type: ignore[arg-type]
            except Exception:
                self._dpo_adapter = None

    def _manifold_feasible(self, causes: List[Any], effect: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Wrapper invoking manifold.check_feasible(causes, effect, context) when enabled.
        Returns True when dual_proof disabled or manifold missing; robust to backend errors.
        """
        dp = getattr(self.cfg, "dual_proof", None)
        if self.manifold is None or dp is None or not getattr(dp, "enable", False):
            return True
        try:
            return bool(self.manifold.check_feasible(causes, effect, context))
        except Exception:
            # Be permissive on backend errors to preserve robustness
            return True

    def _check_hyperpath_feasible(self, hyperpath: Hyperpath) -> bool:
        """
        Minimal policy: return True only if all edges in hyperpath pass manifold feasibility.
        When disabled, returns True. Robust to missing vertices/edges and backend errors.
        """
        if not self._dual_proof_enabled("backward"):
            return True
        try:
            for eid in hyperpath.edges:
                e = self.hypergraph.get_edge(eid)
                if e is None:
                    continue
                effect_v = self.hypergraph.get_vertex(e.head)
                if effect_v is None:
                    continue
                tails_v = [self.hypergraph.get_vertex(vid) for vid in sorted(e.tail)]
                if not all(tv is not None for tv in tails_v):
                    continue
                if not self._manifold_feasible(tails_v, effect_v, context={"check_point": "backward"}):
                    return False
            return True
        except Exception:
            return True

    @no_grad_decorator
    def step(
        self,
        events: Sequence[Event],
        *,
        # Optional supervised targets for credit assignment (list of VertexIds to traverse from)
        target_vertices: Optional[Sequence[VertexId]] = None,
        sign: int = +1,
        freeze_plasticity: bool = False,
    ) -> Mapping[str, Any]:
        """
        Process one batch of events.

        Returns a metrics mapping including:
        - n_events_ingested, n_vertices_new
        - n_candidates, n_admitted
        - n_hyperpaths, n_edges_updated, n_pruned
        """
        metrics: Dict[str, Any] = {
            "n_events_ingested": 0,
            "n_vertices_new": 0,
            "n_candidates": 0,
            "n_admitted": 0,
            "n_hyperpaths": 0,
            "n_edges_updated": 0,
            "n_pruned": 0,
            # Dual-proof gating counters (zero-initialized)
            "n_grow_rejected_manifold": 0,
            "n_grow_nonfeasible": 0,
            "n_backward_rejected_manifold": 0,
            "n_backward_nonfeasible": 0,
        }
        # ISO8601 timestamp for audit emissions in this step (wall-clock)
        now_t_iso = datetime.utcnow().isoformat() + "Z"

        # 1) Ingest events and collect newly materialized vertices ordered by time
        new_vertices: List[Vertex] = []
        for ev in events:
            v = self.hypergraph.ingest_event(ev)
            new_vertices.append(v)
        new_vertices.sort(key=lambda v: v.t)
        metrics["n_events_ingested"] = len(events)
        metrics["n_vertices_new"] = len(new_vertices)

        # 2) Per new head vertex, generate and admit DHG candidates
        for head in new_vertices:
            t_head = head.t
            w: Window = (t_head - self.cfg.dhg.delay_max, t_head - self.cfg.dhg.delay_min)
            cand = self.dhg.generate_candidates_tc_knn(
                hypergraph=self.hypergraph,
                connectivity=self.connectivity,
                head_vertex=head,
                window=w,
                k=self.cfg.dhg.k,
                combination_order_max=self.cfg.dhg.combination_order_max,
                causal_coincidence_delta=self.cfg.dhg.causal_coincidence_delta,
                budget_per_head=self.cfg.dhg.budget_per_head,
                init_reliability=self.cfg.dhg.init_reliability,
                refractory_rho=self.cfg.dhg.refractory_rho,
            )
            metrics["n_candidates"] += len(cand)

            # Dual-proof GROW acceptance gating (optional)
            cand_to_admit: Sequence[Any] = cand
            if self._dual_proof_enabled("grow") and cand:
                mode = self._dp_mode()
                _filtered: List[Any] = []
                for e in cand:
                    try:
                        effect_v = self.hypergraph.get_vertex(e.head)
                        tails_v = [self.hypergraph.get_vertex(vid) for vid in sorted(e.tail)]
                        # Only call manifold when vertices are resolvable
                        if effect_v is not None and all(tv is not None for tv in tails_v):
                            feasible = self._manifold_feasible(tails_v, effect_v, context={"check_point": "grow"})
                        else:
                            feasible = True  # permissive to avoid unintended rejections
                    except Exception:
                        feasible = True
                    if not feasible:
                        if mode == "hard":
                            metrics["n_grow_rejected_manifold"] += 1
                            # skip add
                            continue
                        else:
                            metrics["n_grow_nonfeasible"] += 1
                    _filtered.append(e)
                cand_to_admit = _filtered

            admitted_ids: List[str] = []
            if self._dpo_enabled() and ("grow" in self._dpo_apply_ops()) and cand_to_admit:
                try:
                    self._ensure_dpo()
                    added_ids: List[str] = []
                    for e in cand_to_admit:
                        try:
                            rule = DPO_Rule(
                                name="grow",
                                kind="GROW",
                                lkr=DPO_LKR(),
                                preconditions={},
                                params={"tails": list(e.tail), "head": e.head, "attributes": {"frozen": False}},
                            )
                            res = self._dpo_engine.apply(rule, DPO_Match(), self._dpo_adapter)  # type: ignore[arg-type]
                            if getattr(res, "applied", False) and res.changes.get("added_edges"):
                                added_ids.extend([str(x) for x in res.changes.get("added_edges", [])])
                        except Exception:
                            # Skip candidate on failure; rely on legacy fallback if nothing added
                            pass
                    metrics["n_admitted"] += len(added_ids)
                    admitted_ids = added_ids
                    if not admitted_ids:
                        # Fallback to legacy if none added via DPO
                        admitted = self.dhg.admit(self.hypergraph, cand_to_admit)
                        metrics["n_admitted"] += len(admitted)
                        admitted_ids = [str(eid) for eid in admitted]
                except Exception:
                    admitted = self.dhg.admit(self.hypergraph, cand_to_admit)
                    metrics["n_admitted"] += len(admitted)
                    admitted_ids = [str(eid) for eid in admitted]
            else:
                admitted = self.dhg.admit(self.hypergraph, cand_to_admit)
                metrics["n_admitted"] += len(admitted)
                admitted_ids = [str(eid) for eid in admitted]
            # EAT: emit GROW for admitted edges (if configured)
            if getattr(self, "_audit", None) and admitted_ids:
                try:
                    self._audit.emit_grow(admitted_ids, now_t_iso)
                except Exception:
                    pass

        # 3) Optional credit assignment via backward traversal from target vertices
        if target_vertices:
            hp_total = 0
            hp_list_all: List[Hyperpath] = []
            for vid in target_vertices:
                v = self.hypergraph.get_vertex(vid)
                if v is None:
                    continue
                # Traversal
                hp_list = self.traversal.backward_traverse(
                    hypergraph=self.hypergraph,
                    target=v,
                    horizon=self.cfg.traversal.horizon,
                    beam_size=self.cfg.traversal.beam_size,
                    rng=None,
                    refractory_enforce=True,
                )
                # Dual-proof BACKWARD gating
                if self._dual_proof_enabled("backward") and hp_list:
                    mode = self._dp_mode()
                    _kept: List[Hyperpath] = []
                    for _hp in hp_list:
                        try:
                            ok = self._check_hyperpath_feasible(_hp)
                        except Exception:
                            ok = True
                        if not ok:
                            if mode == "hard":
                                metrics["n_backward_rejected_manifold"] += 1
                                # drop this hyperpath
                                continue
                            else:
                                metrics["n_backward_nonfeasible"] += 1
                        _kept.append(_hp)
                    hp_list = _kept

                hp_total += len(hp_list)
                hp_list_all.extend(hp_list)
            metrics["n_hyperpaths"] = hp_total
            # EAT: emit EAT records for all traversed hyperpaths (if configured)
            if getattr(self, "_audit", None) and hp_list_all:
                try:
                    now_t_us = new_vertices[-1].t if new_vertices else 0
                    for _hp in hp_list_all:
                        self._audit.emit_eat(_hp, now_t_us)
                except Exception:
                    pass

            # FSM observe + structural abstraction (feature-gated)
            if self.cfg.enable_abstraction and self.fsm_engine is not None and self.abstraction_engine is not None:
                label_to_hp: Dict[str, Hyperpath] = {}
                observed_for_fsm: List[Hyperpath] = []
                for hp in hp_list_all:
                    # Canonical label: WL canonicalizer if available; else traversal/fallback
                    if self._wl_canonizer is not None:
                        canon_label = self._wl_canonizer.canonical_label(hp)
                    else:
                        canon_label = hp.label if hp.label is not None else "|".join(sorted([str(e) for e in hp.edges]))
                    label_to_hp[canon_label] = hp
                    observed_for_fsm.append(
                        Hyperpath(
                            head=hp.head,
                            edges=hp.edges,
                            score=hp.score,
                            length=hp.length,
                            label=canon_label,
                        )
                    )

                # Observe all hyperpaths for this step
                _ = self.fsm_engine.observe(observed_for_fsm, now_t=new_vertices[-1].t if new_vertices else 0)

                # Query promotions since last call and cap per-step work
                promo_labels = list(self.fsm_engine.pop_promotions())
                limit = int(getattr(self.cfg, "fsm_promotion_limit_per_step", getattr(self.cfg.fsm, "promotion_limit_per_step", 1)))
                promoted = 0
                for lbl in promo_labels:
                    if promoted >= limit:
                        break
                    hp_inst = label_to_hp.get(lbl)
                    if hp_inst is None:
                        continue
                    try:
                        self.abstraction_engine.promote(hp_inst)
                        promoted += 1
                        # CRC extraction + logging (optional, non-fatal)
                        if getattr(self, "_crc_extractor", None) is not None:
                            try:
                                try:
                                    support_scalar = float(getattr(self.cfg.fsm, "theta", 1.0))
                                except Exception:
                                    support_scalar = 1.0
                                card = self._crc_extractor.make_card(lbl, hp_inst, support=support_scalar)
                                if getattr(self, "_crc_logger", None) is not None:
                                    try:
                                        self._crc_logger.append(card)
                                    except Exception:
                                        pass
                            except Exception:
                                # Do not affect main pipeline on CRC failures
                                pass
                    except Exception:
                        # Keep pipeline robust; continue other candidates
                        pass

                metrics["n_fsm_observed"] = len(observed_for_fsm)
                metrics["n_fsm_promoted"] = promoted
                metrics["n_abstracted"] = promoted

            # Plasticity update + prune
            pstate = self.cfg.plasticity.to_state(freeze=freeze_plasticity)
            updated = self.plasticity.update_from_evidence(
                hypergraph=self.hypergraph,
                hyperpaths=hp_list_all,
                sign=sign,
                now_t=new_vertices[-1].t if new_vertices else 0,
                state=pstate,
            )
            metrics["n_edges_updated"] = len(updated)
            # EAT: emit UPDATE for reliability changes (if configured)
            if getattr(self, "_audit", None) and updated:
                try:
                    self._audit.emit_update(updated, now_t_iso)
                except Exception:
                    pass

            if self._dpo_enabled() and ("prune" in self._dpo_apply_ops()):
                try:
                    self._ensure_dpo()
                    snap = self.hypergraph.snapshot()
                    theta = float(getattr(getattr(self.cfg, "dpo", None), "theta_prune", getattr(self.cfg.plasticity, "prune_threshold", 0.05)))
                    removed = 0
                    for eid in list(snap.hyperedges.keys()):
                        try:
                            r = float(self._dpo_adapter.get_reliability(eid))  # type: ignore[union-attr]
                        except Exception:
                            e_obj = self.hypergraph.get_edge(eid)
                            r = float(e_obj.reliability) if e_obj is not None else 1.0
                        if r <= theta:
                            try:
                                rule = DPO_Rule(
                                    name="prune",
                                    kind="PRUNE",
                                    lkr=DPO_LKR(),
                                    preconditions={"theta_prune": theta},
                                    params={"edge_id": eid, "theta_prune": theta},
                                )
                                res = self._dpo_engine.apply(rule, DPO_Match(edge_id=eid), self._dpo_adapter)  # type: ignore[arg-type]
                                if getattr(res, "applied", False):
                                    removed += 1
                            except Exception:
                                # Continue on per-edge failure
                                pass
                    metrics["n_pruned"] = removed
                    if removed == 0:
                        pruned = self.plasticity.prune(
                            hypergraph=self.hypergraph,
                            now_t=new_vertices[-1].t if new_vertices else 0,
                            state=pstate,
                        )
                        metrics["n_pruned"] = pruned
                except Exception:
                    pruned = self.plasticity.prune(
                        hypergraph=self.hypergraph,
                        now_t=new_vertices[-1].t if new_vertices else 0,
                        state=pstate,
                    )
                    metrics["n_pruned"] = pruned
            else:
                pruned = self.plasticity.prune(
                    hypergraph=self.hypergraph,
                    now_t=new_vertices[-1].t if new_vertices else 0,
                    state=pstate,
                )
                metrics["n_pruned"] = pruned

        # Optional SNN forward hook (no effect on DCH metrics)
        if getattr(self, "_snn_enabled", False) and self.encoder is not None and events:
            try:
                import importlib
                torch = importlib.import_module("torch")
                device = torch.device(self._snn_device if (torch.cuda.is_available() or "cpu" in str(self._snn_device)) else "cpu")
                t0 = min(int(e.t) for e in events)
                t1 = max(int(e.t) for e in events)
                spikes, meta = self.encoder.encode(events, (t0, t1), device)
                N = int(meta.get("N", 0))
                T = int(meta.get("T", 0))
                if N > 0 and T > 0:
                    if self.snn_model is None or self._snn_input_size != N:
                        # Resolve model config with input size
                        cfg_map = dict(self._snn_model_config)
                        model_map = dict(cfg_map.get("model", {}))
                        topo_map = dict(cfg_map.get("topology", {}))
                        topo_map["input_size"] = N
                        if "num_classes" not in topo_map or int(topo_map.get("num_classes", 0)) <= 0:
                            topo_map["num_classes"] = N
                        cfg_map["model"] = model_map
                        cfg_map["topology"] = topo_map
                        # Build model lazily
                        norse_models = importlib.import_module("dch_snn.norse_models")
                        self.snn_model, self.snn_meta = norse_models.create_model(cfg_map)
                        self.snn_model = self.snn_model.to(device)
                        self.snn_model.eval()
                        self._snn_input_size = N
                    # Forward
                    logits, aux = self.snn_model(spikes)
                    try:
                        mval = float(torch.mean(torch.abs(logits)).item())
                    except Exception:
                        mval = 0.0
                    metrics["snn_T"] = T
                    metrics["snn_N"] = N
                    metrics["snn_forward"] = 1
                    metrics["snn_logits_mean_abs"] = mval
                else:
                    metrics["snn_T"] = T
                    metrics["snn_N"] = N
                    metrics["snn_forward"] = 0
            except ImportError:
                # Should not occur given early gating; ignore to keep DCH-only path robust
                metrics["snn_forward"] = 0
            except Exception:
                # Keep robust; do not fail pipeline due to SNN issues
                metrics["snn_forward"] = 0

        return metrics

    # -------------------------
    # Convenience constructor
    # -------------------------

    @classmethod
    def from_defaults(
        cls,
        *,
        cfg: PipelineConfig = PipelineConfig(),
        connectivity_map: Optional[Mapping[int, Iterable[int]]] = None,
        encoder_config: Optional[EncoderConfig] = None,
    ) -> Tuple["DCHPipeline", SimpleBinnerEncoder]:
        """
        Build a usable pipeline with in-memory storage and default engines.

        Returns:
            pipeline: DCHPipeline
            encoder: SimpleBinnerEncoder
        """
        hypergraph = InMemoryHypergraph()
        connectivity = StaticGraphConnectivity(connectivity_map or {})
        dhg = DefaultDHGConstructor()
        traversal = DefaultTraversalEngine(length_penalty_base=cfg.traversal.length_penalty_base)
        # Select plasticity engine based on config knob (non-breaking default = "ema")
        impl = getattr(cfg.plasticity, "impl", "ema")
        if impl not in {"ema", "beta"}:
            raise ValueError(f"Invalid plasticity.impl='{impl}'. Allowed values are {{'ema','beta'}}.")
        if impl == "beta":
            # Local import to avoid import cost when not used
            from dch_core.plasticity_beta import BetaPlasticityEngine
            plasticity = BetaPlasticityEngine(alpha0=1.0, beta0=1.0)
        else:
            plasticity = DefaultPlasticityEngine()
        pipeline = cls(
            hypergraph=hypergraph,
            connectivity=connectivity,
            dhg=dhg,
            traversal=traversal,
            plasticity=plasticity,
            cfg=cfg,
        )

        # Optional FSM + Abstraction wiring (torch-free, in-memory)
        if cfg.enable_abstraction:
            fsm = StreamingFSMEngine(
                theta=cfg.fsm.theta,
                lambda_decay=cfg.fsm.lambda_decay,
                hold_k=cfg.fsm.hold_k,
                min_weight=cfg.fsm.min_weight,
            )
            pipeline.fsm_engine = fsm
            pipeline.abstraction_engine = DefaultAbstractionEngine(
                hypergraph,
                AbstractionParams(**dict(cfg.abstraction_params or {})),
            )
            # WL canonicalizer for stable, permutation/time-shift invariant labels
            pipeline._wl_canonizer = WLHyperpathEmbedding(WLParams())
            # Optional CRC wiring (opt-in; non-fatal)
            try:
                if getattr(cfg, "crc_log_path", None):
                    # Local imports to keep optional
                    from dch_core.crc import CRCExtractor as _CRCExtractor
                    from dch_pipeline.crc_logger import CRCLogger as _CRCLogger
                    pipeline._crc_extractor = _CRCExtractor(hypergraph, alpha0=1.0, beta0=1.0)
                    pipeline._crc_logger = _CRCLogger(str(cfg.crc_log_path))
            except Exception:
                pipeline._crc_extractor = None
                pipeline._crc_logger = None

        enc = SimpleBinnerEncoder(encoder_config or EncoderConfig())
        pipeline.encoder = enc

        # Optional EAT audit logger wiring (env var DCH_EAT_LOG or cfg.audit_log_path)
        log_path = getattr(cfg, "audit_log_path", None) or os.getenv("DCH_EAT_LOG")
        if log_path:
            try:
                from dch_pipeline.eat_logger import EATAuditLogger
                pipeline._audit = EATAuditLogger(log_path)
            except Exception:
                # Non-fatal wiring; keep pipeline operational
                pipeline._audit = None

        # Optional manifold wiring (feature-gated; non-enforcing in P2-11)
        try:
            mcfg = getattr(cfg, "manifold", None)
            if mcfg is not None and getattr(mcfg, "enable", False):
                impl = str(getattr(mcfg, "impl", "noop")).lower()
                if impl == "noop":
                    try:
                        from dch_core.manifold import NoOpManifold
                        pipeline.manifold = NoOpManifold()
                        try:
                            import logging
                            logging.getLogger(__name__).info(
                                "Manifold enabled: %s v%s",
                                pipeline.manifold.name(),
                                pipeline.manifold.version(),
                            )
                        except Exception:
                            pass
                    except Exception:
                        pipeline.manifold = None
                else:
                    # Unknown impl: leave disabled (no behavior changes in this task)
                    pipeline.manifold = None
            else:
                pipeline.manifold = None
        except Exception:
            pipeline.manifold = None

        # Optional SNN gating (lazy; model constructed on first forward when input size is known)
        try:
            snn_cfg = getattr(cfg, "snn", None)
            if snn_cfg and getattr(snn_cfg, "enabled", False):
                import importlib.util as _ilu
                missing = []
                if _ilu.find_spec("torch") is None:
                    missing.append("torch")
                if _ilu.find_spec("norse") is None:
                    missing.append("norse")
                if missing:
                    missing_str = ", ".join(missing)
                    raise ImportError(
                        f"SNN enabled but missing optional dependency/dependencies: {missing_str}. "
                        "Install with:\n"
                        "- pip install 'torch>=2.2' 'norse>=0.0.9'\n"
                        "- or conda install -c conda-forge pytorch norse\n"
                        "Alternatively set snn.enabled=false in your config."
                    )
                pipeline._snn_enabled = True
                pipeline._snn_device = getattr(snn_cfg, "device", "cpu")
                # Base model config; topology.input_size resolved at first forward based on encoder meta['N']
                pipeline._snn_model_config = {
                    "model": {"name": getattr(snn_cfg, "model", "norse_lif"), "unroll": bool(getattr(snn_cfg, "unroll", True))},
                }
                # Merge additional parameters if provided
                extra_params = getattr(snn_cfg, "model_params", {})
                if isinstance(extra_params, dict) and extra_params:
                    pipeline._snn_model_config.update(extra_params)
        except Exception:
            # Keep pipeline import-safe: re-raise to caller for clarity
            raise

        return pipeline, enc


__all__ = [
    "DHGConfig",
    "TraversalConfig",
    "PlasticityConfig",
    "FSMConfig",
    "SNNConfig",
    "ManifoldConfig",
    "DualProofConfig",
    "PipelineConfig",
    "DCHPipeline",
]