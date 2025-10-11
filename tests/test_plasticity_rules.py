# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Plasticity rules unit tests (torch-free).

Covers:
- Potentiation (sign=+1): reliability increases by alpha * s_norm and clamps to [lo, hi].
- Depression (sign=-1): reliability decreases by alpha * s_norm and clamps to [lo, hi].
- Success/miss counters increment appropriately.
- Prune removes edges strictly below threshold.

Runtime: < 100ms.
"""

from __future__ import annotations

import math
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
from dch_core.plasticity import DefaultPlasticityEngine


def _make_edge(hg: InMemoryHypergraph, head_t: int, tail_dt: int, reliability: float, tag: str) -> Hyperedge:
    """
    Construct a unary edge with head at head_t and tail at (head_t - tail_dt).
    Edge window delta_min/max bracket the exact tail delay to ensure admissibility.
    """
    head_vid = make_vertex_id(10, head_t)
    tail_vid = make_vertex_id(1 + (hash(tag) % 1000), head_t - tail_dt)  # different tail per tag to avoid dedup
    # Materialize vertices
    hg.ingest_event(Event(neuron_id=10, t=head_t))
    hg.ingest_event(Event(neuron_id=int(str(tail_vid).split("@")[0]), t=head_t - tail_dt))
    # Temporal window matches observed delay
    dmin, dmax = tail_dt, tail_dt
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


def test_potentiation_increases_and_clamps():
    hg = InMemoryHypergraph()
    e = _make_edge(hg, head_t=1000, tail_dt=100, reliability=0.50, tag="pot1")
    _ = hg.insert_hyperedges([e])

    plast = DefaultPlasticityEngine()
    state = PlasticityState(ema_alpha=0.10, reliability_clamp=(0.02, 0.98), prune_threshold=0.05)

    # Evidence: single hyperpath containing edge 'e' with positive score
    hp = Hyperpath(head=e.head, edges=(e.id,), score=1.0, length=1, label="e")
    updates = plast.update_from_evidence(hypergraph=hg, hyperpaths=[hp], sign=+1, now_t=1001, state=state)

    # r_new = r_old + alpha * 1.0 (since s_norm=1 for single-edge evidence)
    assert updates[e.id] == pytest.approx(0.60, rel=1e-9, abs=1e-12)
    assert hg.get_edge(e.id).reliability == pytest.approx(0.60, rel=1e-9, abs=1e-12)
    assert hg.get_edge(e.id).counts_success == 1
    assert hg.get_edge(e.id).counts_miss == 0

    # Clamp at upper bound
    hg.get_edge(e.id).reliability = 0.979
    state2 = PlasticityState(ema_alpha=0.50, reliability_clamp=(0.02, 0.98), prune_threshold=0.05)
    updates2 = plast.update_from_evidence(hypergraph=hg, hyperpaths=[hp], sign=+1, now_t=1002, state=state2)
    assert updates2[e.id] == pytest.approx(0.98, rel=1e-12, abs=0.0)  # exact clamp to hi
    assert hg.get_edge(e.id).reliability == pytest.approx(0.98, rel=1e-12, abs=0.0)
    assert hg.get_edge(e.id).counts_success == 2  # incremented again


def test_depression_decreases_and_counts_miss_and_prune():
    hg = InMemoryHypergraph()
    e_main = _make_edge(hg, head_t=2000, tail_dt=150, reliability=0.80, tag="dep-main")
    e_low = _make_edge(hg, head_t=2100, tail_dt=120, reliability=0.04, tag="dep-low")
    admitted = hg.insert_hyperedges([e_main, e_low])
    assert set(admitted) == {e_main.id, e_low.id}

    plast = DefaultPlasticityEngine()
    state = PlasticityState(ema_alpha=0.20, reliability_clamp=(0.02, 0.98), prune_threshold=0.05)

    # Negative evidence on main edge
    hp = Hyperpath(head=e_main.head, edges=(e_main.id,), score=1.0, length=1, label="e_main")
    updates = plast.update_from_evidence(hypergraph=hg, hyperpaths=[hp], sign=-1, now_t=2001, state=state)

    # r_new = 0.80 - 0.20 = 0.60
    assert updates[e_main.id] == pytest.approx(0.60, rel=1e-9, abs=1e-12)
    e_after = hg.get_edge(e_main.id)
    assert e_after is not None
    assert e_after.reliability == pytest.approx(0.60, rel=1e-9, abs=1e-12)
    assert e_after.counts_miss == 1
    assert e_after.counts_success == 0

    # Prune removes edges with reliability < threshold (e_low at 0.04)
    pruned = plast.prune(hg, now_t=2002, state=state)
    assert pruned >= 1
    assert hg.get_edge(e_low.id) is None