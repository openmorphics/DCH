# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
P2-12 Dual-proof gating (causal + manifold): tests for feature-flagged gating in GROW acceptance
and backward checks.

Deterministic, offline, and quick tests using the same two-event synthetic pattern used across smoke tests.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pytest

from dch_core.interfaces import Event, make_vertex_id
from dch_pipeline.evaluation import summarize_reliability
from dch_pipeline.pipeline import (
    DCHPipeline,
    PipelineConfig,
    DHGConfig,
    ManifoldConfig,
    DualProofConfig,
)


def _build_events_and_target(cfg: PipelineConfig) -> Tuple[list[Event], str]:
    """Deterministic 2-event sequence: one presynaptic event then a head event at midpoint delay."""
    # Defaults: small graph with one connection 1 -> 2
    head_neuron = 2
    tail_neuron = 1
    t_head = 10_000
    dmin = int(cfg.dhg.delay_min)
    dmax = int(cfg.dhg.delay_max)
    mid_delay = int((dmin + dmax) // 2)
    if mid_delay < dmin:
        mid_delay = dmin
    if mid_delay > dmax:
        mid_delay = dmax
    t_presyn = int(t_head - mid_delay)

    events = [
        Event(neuron_id=tail_neuron, t=t_presyn, meta=None),
        Event(neuron_id=head_neuron, t=t_head, meta=None),
    ]
    target_vid = make_vertex_id(head_neuron, t_head)
    return events, target_vid


def _edge_count(pipeline: DCHPipeline) -> int:
    return len(pipeline.hypergraph.snapshot().hyperedges)


class DummyManifold:
    """A deterministic manifold backend that declares all inputs non-feasible (False)."""
    def name(self) -> str:
        return "dummy"

    def version(self) -> str:
        return "1.0"

    def serialize_config(self) -> Dict[str, Any]:
        return {"type": "dummy"}

    def check_feasible(self, causes: list, effect: Any, context: Dict[str, Any] | None = None) -> bool:
        return False

    def explain(self, causes: list, effect: Any, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return {"type": "dummy", "feasible": False}


def test_dual_proof_default_off_parity():
    """
    Default OFF parity:
    - Build two pipelines: one with default config (dual_proof disabled implicitly)
      vs explicit dual_proof=DualProofConfig(enable=False).
    - Run minimal deterministic step and assert step metrics and reliability summaries are equal.
    """
    conn = {2: [1]}
    base_cfg = PipelineConfig()
    off_cfg = PipelineConfig(dual_proof=DualProofConfig(enable=False))

    # Pipeline A: default
    pA, _ = DCHPipeline.from_defaults(cfg=base_cfg, connectivity_map=conn)
    evA, tgtA = _build_events_and_target(base_cfg)
    mA = pA.step(evA, target_vertices=[tgtA], sign=+1)
    sA = summarize_reliability(pA.hypergraph, alpha0=1.0, beta0=1.0, ci_level=0.95, max_edges=10)

    # Pipeline B: explicit OFF
    pB, _ = DCHPipeline.from_defaults(cfg=off_cfg, connectivity_map=conn)
    evB, tgtB = _build_events_and_target(off_cfg)
    mB = pB.step(evB, target_vertices=[tgtB], sign=+1)
    sB = summarize_reliability(pB.hypergraph, alpha0=1.0, beta0=1.0, ci_level=0.95, max_edges=10)

    assert mA == mB
    assert sA == sB


def test_dual_proof_noop_soft_parity():
    """
    NoOp manifold parity in soft mode:
    - Enable manifold NoOp and dual_proof soft.
    - Since NoOp returns True, behavior unchanged and gating counters remain zero.
    """
    conn = {2: [1]}
    base_cfg = PipelineConfig()
    soft_cfg = PipelineConfig(
        manifold=ManifoldConfig(enable=True, impl="noop"),
        dual_proof=DualProofConfig(enable=True, mode="soft"),
    )

    # Baseline pipeline (no gating)
    p0, _ = DCHPipeline.from_defaults(cfg=base_cfg, connectivity_map=conn)
    ev0, tgt0 = _build_events_and_target(base_cfg)
    m0 = p0.step(ev0, target_vertices=[tgt0], sign=+1)
    s0 = summarize_reliability(p0.hypergraph, alpha0=1.0, beta0=1.0, ci_level=0.95, max_edges=10)

    # Soft-gated pipeline with NoOp manifold
    p1, _ = DCHPipeline.from_defaults(cfg=soft_cfg, connectivity_map=conn)
    ev1, tgt1 = _build_events_and_target(soft_cfg)
    m1 = p1.step(ev1, target_vertices=[tgt1], sign=+1)
    s1 = summarize_reliability(p1.hypergraph, alpha0=1.0, beta0=1.0, ci_level=0.95, max_edges=10)

    # Parity on metrics and reliability summary
    assert m0 == m1
    assert s0 == s1
    # Gating counters remain zero
    assert m1.get("n_grow_rejected_manifold", 0) == 0
    assert m1.get("n_grow_nonfeasible", 0) == 0
    assert m1.get("n_backward_rejected_manifold", 0) == 0
    assert m1.get("n_backward_nonfeasible", 0) == 0


def test_dual_proof_grow_hard_rejects():
    """
    GROW hard mode rejects:
    - Enable dual_proof=hard with check_points=["grow"].
    - Monkeypatch pipeline.manifold with DummyManifold that always returns False.
    - Run a step and assert:
        * metrics["n_grow_rejected_manifold"] >= 1
        * hypergraph admitted fewer edges than a control pipeline without gating.
    """
    conn = {2: [1]}
    # Control (no gating)
    ctrl_cfg = PipelineConfig()
    p_ctrl, _ = DCHPipeline.from_defaults(cfg=ctrl_cfg, connectivity_map=conn)
    evC, tgtC = _build_events_and_target(ctrl_cfg)
    m_ctrl = p_ctrl.step(evC, target_vertices=[tgtC], sign=+1)
    ctrl_edges = _edge_count(p_ctrl)
    assert m_ctrl.get("n_admitted", 0) >= 1
    assert ctrl_edges >= 1

    # Gating pipeline
    grow_cfg = PipelineConfig(
        manifold=ManifoldConfig(enable=True, impl="noop"),
        dual_proof=DualProofConfig(enable=True, mode="hard", check_points=("grow",)),
    )
    p_grow, _ = DCHPipeline.from_defaults(cfg=grow_cfg, connectivity_map=conn)
    # Monkeypatch manifold to reject everything
    p_grow.manifold = DummyManifold()
    evG, tgtG = _build_events_and_target(grow_cfg)
    m_grow = p_grow.step(evG, target_vertices=[tgtG], sign=+1)
    grow_edges = _edge_count(p_grow)

    assert m_grow.get("n_grow_rejected_manifold", 0) >= 1
    # Should admit fewer edges than control (likely zero)
    assert grow_edges < ctrl_edges


def test_dual_proof_backward_soft_counts():
    """
    Backward soft mode counts:
    - Enable dual_proof soft with check_points=["backward"].
    - Monkeypatch pipeline.manifold with DummyManifold (always False).
    - Trigger backward traversal and assert metrics["n_backward_nonfeasible"] >= 1.
    """
    conn = {2: [1]}
    bw_cfg = PipelineConfig(
        manifold=ManifoldConfig(enable=True, impl="noop"),
        dual_proof=DualProofConfig(enable=True, mode="soft", check_points=("backward",)),
    )
    p_bw, _ = DCHPipeline.from_defaults(cfg=bw_cfg, connectivity_map=conn)
    p_bw.manifold = DummyManifold()
    evB, tgtB = _build_events_and_target(bw_cfg)
    m_bw = p_bw.step(evB, target_vertices=[tgtB], sign=+1)

    assert m_bw.get("n_backward_nonfeasible", 0) >= 1