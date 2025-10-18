# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Beta–Bernoulli plasticity engine tests (torch-free and fast).

Covers:
- Positive evidence increases reliability toward 1 via posterior mean with priors.
- Negative evidence decreases reliability toward 0 via posterior mean with priors.
- Prune delegates to hypergraph threshold.
- Pipeline config knob selects Beta engine (non-breaking default remains EMA).

Acceptance:
- Posterior mean matches: rho = (alpha0 + s) / (alpha0 + beta0 + s + f)
- Tolerances at 1e-6.
"""

from __future__ import annotations

import pytest

from dch_core.interfaces import (
    EdgeId,
    Event,
    Hyperedge,
    Hyperpath,
    PlasticityState,
    make_edge_id,
    make_vertex_id,
)
from dch_core.hypergraph_mem import InMemoryHypergraph
from dch_core.plasticity_beta import BetaPlasticityEngine

# Pipeline selection test imports
from dch_pipeline.pipeline import DCHPipeline, PipelineConfig, PlasticityConfig


def _make_unary_edge(
    hg: InMemoryHypergraph,
    *,
    head_t: int,
    tail_dt: int,
    reliability: float,
    tag: str,
) -> Hyperedge:
    """
    Construct a unary admissible edge for tests:
    - Materializes head at head_t and tail at (head_t - tail_dt)
    - Sets delta_min/max to the exact delay to ensure admissibility
    """
    head_vid = make_vertex_id(10, head_t)
    # Ensure unique tail neuron per tag to avoid dedup collisions
    tail_neuron = 1 + (hash(tag) % 1000003)
    tail_vid = make_vertex_id(tail_neuron, head_t - tail_dt)
    # Materialize vertices
    hg.ingest_event(Event(neuron_id=10, t=head_t))
    hg.ingest_event(Event(neuron_id=tail_neuron, t=head_t - tail_dt))
    # Edge window matches observed delay
    dmin = dmax = int(tail_dt)
    eid = make_edge_id(head=head_vid, tail={tail_vid}, t=head_t)
    return Hyperedge(
        id=eid,
        tail={tail_vid},
        head=head_vid,
        delta_min=dmin,
        delta_max=dmax,
        refractory_rho=0,
        reliability=float(reliability),
        provenance=tag,
    )


def test_beta_updates_increase_on_positive():
    hg = InMemoryHypergraph()
    e = _make_unary_edge(hg, head_t=1000, tail_dt=100, reliability=0.50, tag="beta-pos")
    admitted = hg.insert_hyperedges([e])
    assert admitted == [e.id]

    plast = BetaPlasticityEngine(alpha0=1.0, beta0=1.0)
    state = PlasticityState(ema_alpha=0.10, reliability_clamp=(0.02, 0.98), prune_threshold=0.05)

    # Single hyperpath evidence containing only this edge with score 1.0
    hp = Hyperpath(head=e.head, edges=(e.id,), score=1.0, length=1, label="e")
    updates = plast.update_from_evidence(hypergraph=hg, hyperpaths=[hp], sign=+1, now_t=1001, state=state)

    # Posterior mean with priors: successes=1, failures=0 -> (1+1)/(1+1+1+0)=2/3
    assert e.id in updates
    assert updates[e.id] == pytest.approx(2.0 / 3.0, rel=1e-9, abs=1e-12)
    e_after = hg.get_edge(e.id)
    assert e_after is not None
    assert e_after.reliability == pytest.approx(2.0 / 3.0, rel=1e-9, abs=1e-12)
    # Counters increment as integers (rounded); single-edge evidence -> +1 success
    assert e_after.counts_success == 1
    assert e_after.counts_miss == 0
    # Increased relative to prior reliability
    assert e_after.reliability > 0.50


def test_beta_updates_decrease_on_negative():
    hg = InMemoryHypergraph()
    e = _make_unary_edge(hg, head_t=2000, tail_dt=120, reliability=0.80, tag="beta-neg")
    _ = hg.insert_hyperedges([e])

    plast = BetaPlasticityEngine(alpha0=1.0, beta0=1.0)
    state = PlasticityState(ema_alpha=0.10, reliability_clamp=(0.02, 0.98), prune_threshold=0.05)

    hp = Hyperpath(head=e.head, edges=(e.id,), score=1.0, length=1, label="e")
    updates = plast.update_from_evidence(hypergraph=hg, hyperpaths=[hp], sign=-1, now_t=2001, state=state)

    # Posterior mean with priors: successes=0, failures=1 -> (1+0)/(1+1+0+1)=1/3
    assert e.id in updates
    assert updates[e.id] == pytest.approx(1.0 / 3.0, rel=1e-9, abs=1e-12)
    e_after = hg.get_edge(e.id)
    assert e_after is not None
    assert e_after.reliability == pytest.approx(1.0 / 3.0, rel=1e-9, abs=1e-12)
    assert e_after.counts_miss == 1
    assert e_after.counts_success == 0
    # Decreased relative to prior reliability
    assert e_after.reliability < 0.80


def test_prune_delegates_threshold():
    hg = InMemoryHypergraph()
    e_hi = _make_unary_edge(hg, head_t=3000, tail_dt=100, reliability=0.50, tag="hi")
    e_lo = _make_unary_edge(hg, head_t=3100, tail_dt=100, reliability=0.04, tag="lo")
    admitted = hg.insert_hyperedges([e_hi, e_lo])
    assert set(admitted) == {e_hi.id, e_lo.id}

    plast = BetaPlasticityEngine()
    state = PlasticityState(prune_threshold=0.05)

    # Prune should remove e_lo (0.04 < 0.05) and keep e_hi
    pruned = plast.prune(hypergraph=hg, now_t=3101, state=state)
    assert pruned >= 1
    assert hg.get_edge(e_lo.id) is None
    assert hg.get_edge(e_hi.id) is not None


def test_pipeline_selects_beta_engine():
    # Build pipeline with impl="beta" (non-breaking knob)
    cfg = PipelineConfig(plasticity=PlasticityConfig(impl="beta"))
    pipeline, _ = DCHPipeline.from_defaults(cfg=cfg, connectivity_map={10: [1]})

    # Type check
    from dch_core.plasticity_beta import BetaPlasticityEngine as _Beta
    assert isinstance(pipeline.plasticity, _Beta)

    # Apply a minimal update through the selected engine directly
    hg = pipeline.hypergraph
    e = _make_unary_edge(hg, head_t=4000, tail_dt=100, reliability=0.50, tag="pipe-beta")
    _ = hg.insert_hyperedges([e])
    hp = Hyperpath(head=e.head, edges=(e.id,), score=1.0, length=1, label="e")
    pstate = cfg.plasticity.to_state()
    updates = pipeline.plasticity.update_from_evidence(
        hypergraph=hg,
        hyperpaths=[hp],
        sign=+1,
        now_t=4001,
        state=pstate,
    )
    assert updates[e.id] == pytest.approx(2.0 / 3.0, rel=1e-9, abs=1e-12)
    assert hg.get_edge(e.id).reliability == pytest.approx(2.0 / 3.0, rel=1e-9, abs=1e-12)