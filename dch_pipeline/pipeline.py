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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
        }

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
            admitted = self.dhg.admit(self.hypergraph, cand)
            metrics["n_admitted"] += len(admitted)

        # 3) Optional credit assignment via backward traversal from target vertices
        if target_vertices:
            hp_total = 0
            hp_list_all = []
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
                hp_total += len(hp_list)
                hp_list_all.extend(hp_list)
            metrics["n_hyperpaths"] = hp_total

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

            pruned = self.plasticity.prune(
                hypergraph=self.hypergraph,
                now_t=new_vertices[-1].t if new_vertices else 0,
                state=pstate,
            )
            metrics["n_pruned"] = pruned

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

        enc = SimpleBinnerEncoder(encoder_config or EncoderConfig())
        return pipeline, enc


__all__ = [
    "DHGConfig",
    "TraversalConfig",
    "PlasticityConfig",
    "FSMConfig",
    "PipelineConfig",
    "DCHPipeline",
]