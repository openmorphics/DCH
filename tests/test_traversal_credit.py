# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Traversal and B-connectivity unit tests.

Focus:
- When only one antecedent of a higher-order edge is present, backward traversal must not
  traverse that edge (violates B-connectivity).
- When both antecedents are present and temporally admissible, traversal returns at least
  one hyperpath containing that edge, and all traversed edges respect temporal constraints.

Runtime: < 100ms, torch-free.
"""

from __future__ import annotations

import pytest

from dch_core.interfaces import (
    Event,
    EdgeId,
    Hyperedge,
    Hyperpath,
    is_temporally_admissible,
    make_edge_id,
    make_vertex_id,
)
from dch_core.hypergraph_mem import InMemoryHypergraph
from dch_core.traversal import DefaultTraversalEngine


def _build_hoe(head_t: int = 1000):
    """
    Build a single higher-order edge with tail size 2 aimed at a head at time head_t.

    Tail events are designed to be temporally admissible when both are present:
      - tail1: neuron 1 @ t=900  (Δ=100)
      - tail2: neuron 2 @ t=850  (Δ=150)
    Edge window: delta_min=50, delta_max=200
    """
    delta_min, delta_max = 50, 200

    head_vid = make_vertex_id(10, head_t)
    tail1_vid = make_vertex_id(1, head_t - 100)  # 900
    tail2_vid = make_vertex_id(2, head_t - 150)  # 850

    tail_set = {tail1_vid, tail2_vid}
    eid = make_edge_id(head=head_vid, tail=tail_set, t=head_t)

    e = Hyperedge(
        id=eid,
        tail=tail_set,
        head=head_vid,
        delta_min=delta_min,
        delta_max=delta_max,
        refractory_rho=0,
        reliability=0.9,
        provenance="test_hoe",
    )
    return e, head_vid, tail1_vid, tail2_vid, delta_min, delta_max


def test_b_connectivity_requires_all_antecedents():
    """
    Only one antecedent present:
    - The HOE must not be traversed.
    - No returned hyperpath should contain the HOE id.
    """
    hg = InMemoryHypergraph()
    trav = DefaultTraversalEngine()

    # Create HOE and insert
    hoe, head_vid, tail1_vid, tail2_vid, dmin, dmax = _build_hoe()
    _ = hg.insert_hyperedges([hoe])

    # Ingest head and only one tail event (tail2 missing)
    head_v = hg.ingest_event(Event(neuron_id=10, t=1000))
    _ = hg.ingest_event(Event(neuron_id=1, t=900))
    # DO NOT ingest (neuron 2, t=850)

    # Backward traversal from head
    results = trav.backward_traverse(
        hypergraph=hg,
        target=head_v,
        horizon=1000,
        beam_size=4,
        rng=None,
        refractory_enforce=True,
    )

    # None of the hyperpaths should include the HOE (B-connectivity violated)
    assert all(hoe.id not in hp.edges for hp in results), "HOE traversal should be impossible with a missing antecedent"


def test_temporal_admissibility_when_all_antecedents_present():
    """
    Both antecedents present and admissible:
    - At least one hyperpath should include the HOE.
    - All traversed edges in that hyperpath must be temporally consistent with their tails and head.
    """
    hg = InMemoryHypergraph()
    trav = DefaultTraversalEngine()

    # Create HOE and insert
    hoe, head_vid, tail1_vid, tail2_vid, dmin, dmax = _build_hoe()
    _ = hg.insert_hyperedges([hoe])

    # Ingest all required events
    head_v = hg.ingest_event(Event(neuron_id=10, t=1000))
    t1_v = hg.ingest_event(Event(neuron_id=1, t=900))
    t2_v = hg.ingest_event(Event(neuron_id=2, t=850))

    # Sanity: admissibility of the HOE w.r.t. current vertices
    assert is_temporally_admissible([t1_v.t, t2_v.t], head_v.t, hoe.delta_min, hoe.delta_max)

    # Backward traversal
    results = trav.backward_traverse(
        hypergraph=hg,
        target=head_v,
        horizon=1000,
        beam_size=8,
        rng=None,
        refractory_enforce=True,
    )

    # Expect at least one hyperpath containing the HOE
    containing = [hp for hp in results if hoe.id in hp.edges]
    assert len(containing) >= 1, "Expected at least one hyperpath containing the HOE when both antecedents are present"

    # For those hyperpaths, verify temporal consistency of all edges they include
    for hp in containing:
        for eid in hp.edges:
            e = hg.get_edge(eid)
            assert e is not None
            head_vtx = hg.get_vertex(e.head)
            assert head_vtx is not None
            tail_times = []
            for tvid in e.tail:
                tv = hg.get_vertex(tvid)
                assert tv is not None
                tail_times.append(tv.t)
            assert is_temporally_admissible(tail_times, head_vtx.t, e.delta_min, e.delta_max)