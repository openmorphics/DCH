# Copyright (c) 2025 DCH Maintainers
# License: MIT

import math
from typing import Tuple

import pytest

from dch_core.abstraction import AbstractionParams, DefaultAbstractionEngine
from dch_core.hypergraph_mem import InMemoryHypergraph
from dch_core.interfaces import EdgeId, Event, Hyperedge, Hyperpath, Vertex, VertexId, make_edge_id


def _build_chain(
    order: Tuple[str, str] = ("A", "B"),
    r1: float = 0.8,
    r2: float = 0.6,
    t_shift: int = 0,
) -> Tuple[InMemoryHypergraph, Vertex, Vertex, Vertex, Vertex, Hyperedge, Hyperedge, Hyperpath]:
    """
    Build a two-step chain in a fresh in-memory hypergraph:

        {A, B} --e1--> X --e2--> Y

    Returns:
        (hg, vA, vB, vX, vY, e1, e2, hyperpath)
    """
    hg = InMemoryHypergraph()

    # Base timestamps (arbitrary, in integer units)
    tA = 100 + t_shift
    tB = 120 + t_shift
    tX = 200 + t_shift
    tY = 300 + t_shift

    # Ingest events -> vertices
    vA = hg.ingest_event(Event(neuron_id=1, t=tA))
    vB = hg.ingest_event(Event(neuron_id=2, t=tB))
    vX = hg.ingest_event(Event(neuron_id=3, t=tX))
    vY = hg.ingest_event(Event(neuron_id=4, t=tY))

    # e1: {A,B} -> X with temporal window capturing observed delays
    dtA = vX.t - vA.t  # 100
    dtB = vX.t - vB.t  # 80
    delta_min_e1 = min(dtA, dtB)
    delta_max_e1 = max(dtA, dtB)

    # Order the tails per 'order' argument (insertion order is irrelevant; id is canonical)
    tails_map = {"A": vA.id, "B": vB.id}
    t1 = tails_map[order[0]]
    t2 = tails_map[order[1]]
    tail_e1 = {t1, t2}

    e1 = Hyperedge(
        id=make_edge_id(head=vX.id, tail=tail_e1, t=vX.t),
        tail=tail_e1,
        head=vX.id,
        delta_min=delta_min_e1,
        delta_max=delta_max_e1,
        refractory_rho=0,
        reliability=r1,
        provenance="e1",
    )

    # e2: {X} -> Y with exact delay window
    dtX = vY.t - vX.t  # 100
    tail_e2 = {vX.id}
    e2 = Hyperedge(
        id=make_edge_id(head=vY.id, tail=tail_e2, t=vY.t),
        tail=tail_e2,
        head=vY.id,
        delta_min=dtX,
        delta_max=dtX,
        refractory_rho=0,
        reliability=r2,
        provenance="e2",
    )

    # Insert edges
    _ = hg.insert_hyperedges([e1, e2])

    # Hyperpath covering the chain (edge order arbitrary)
    hp = Hyperpath(head=vY.id, edges=(e1.id, e2.id), score=r1 * r2, length=2, label=None)

    return hg, vA, vB, vX, vY, e1, e2, hp


def _find_hoe_to_Y(hg: InMemoryHypergraph, vY: Vertex, vA: Vertex, vB: Vertex) -> Hyperedge:
    """Find the unique HOE with head Y and tail {A,B}."""
    candidates = []
    for eid in hg.get_incoming_edges(vY.id):
        e = hg.get_edge(eid)
        if e is None:
            continue
        if e.head == vY.id and set(e.tail) == {vA.id, vB.id}:
            candidates.append(e)
    assert len(candidates) == 1, f"Expected exactly one HOE A,B->Y, found {len(candidates)}"
    return candidates[0]


def test_promote_creates_higher_order_edge():
    hg, vA, vB, vX, vY, e1, e2, hp = _build_chain(order=("A", "B"), r1=0.8, r2=0.6)
    engine = DefaultAbstractionEngine(hg, AbstractionParams())
    new_eid = engine.promote(hp)

    hoe = _find_hoe_to_Y(hg, vY, vA, vB)
    assert str(hoe.id) == str(new_eid)

    # Reliability initialization uses min by default, clamped by floor
    expected = max(AbstractionParams().reliability_floor, min(float(e1.reliability), float(e2.reliability)))
    assert math.isclose(float(hoe.reliability), float(expected), rel_tol=1e-9, abs_tol=1e-12)

    # Provenance payload includes original edge ids
    prov = hoe.attributes.get("provenance")
    assert isinstance(prov, dict)
    from_edges = set(prov.get("from_edges", []))
    assert from_edges == {str(e1.id), str(e2.id)}
    assert isinstance(prov.get("hyperpath_label"), str) and len(prov["hyperpath_label"]) > 0


def test_promote_is_idempotent_with_dedup():
    hg, vA, vB, vX, vY, e1, e2, hp = _build_chain(order=("A", "B"), r1=0.7, r2=0.5)
    engine = DefaultAbstractionEngine(hg, AbstractionParams(dedup=True))

    eid1 = engine.promote(hp)
    eid2 = engine.promote(hp)
    assert str(eid1) == str(eid2)

    # Only one HOE with {A,B} -> Y exists
    _ = _find_hoe_to_Y(hg, vY, vA, vB)


def test_dedup_tail_permutation():
    # Graph 1 with order (A,B)
    hg1, vA1, vB1, vX1, vY1, e1a, e2a, hp1 = _build_chain(order=("A", "B"), r1=0.9, r2=0.4)
    engine1 = DefaultAbstractionEngine(hg1, AbstractionParams())
    eid_1 = engine1.promote(hp1)

    # Graph 2 with order (B,A) for the first edge; HOE id should be identical (permutation invariance)
    hg2, vA2, vB2, vX2, vY2, e1b, e2b, hp2 = _build_chain(order=("B", "A"), r1=0.9, r2=0.4)
    engine2 = DefaultAbstractionEngine(hg2, AbstractionParams())
    eid_2 = engine2.promote(hp2)

    assert str(eid_1) == str(eid_2)


@pytest.mark.parametrize("agg_mode, r1, r2, expected_fn", [
    ("min", 0.81, 0.36, lambda a, b: min(a, b)),
    ("mean", 0.81, 0.36, lambda a, b: (a + b) / 2.0),
    ("geo", 0.81, 0.36, lambda a, b: (a * b) ** 0.5),
])
def test_reliability_aggregation_modes(agg_mode, r1, r2, expected_fn):
    # Case 1: above floor, direct aggregation
    hg, vA, vB, vX, vY, e1, e2, hp = _build_chain(order=("A", "B"), r1=r1, r2=r2)
    params = AbstractionParams(reliability_agg=agg_mode, reliability_floor=0.1)
    engine = DefaultAbstractionEngine(hg, params)
    eid = engine.promote(hp)
    hoe = hg.get_edge(EdgeId(str(eid)))
    assert hoe is not None
    expected = expected_fn(r1, r2)
    assert math.isclose(float(hoe.reliability), float(expected), rel_tol=1e-9, abs_tol=1e-12)

    # Case 2: below floor, ensure clamping
    hg2, vA2, vB2, vX2, vY2, e1b, e2b, hp2 = _build_chain(order=("A", "B"), r1=0.0, r2=0.05)
    engine2 = DefaultAbstractionEngine(hg2, params)
    eid2 = engine2.promote(hp2)
    hoe2 = hg2.get_edge(EdgeId(str(eid2)))
    assert hoe2 is not None
    assert float(hoe2.reliability) >= float(params.reliability_floor) - 1e-12


def test_time_shift_invariance_of_label():
    # Promote on base times
    hg1, vA1, vB1, vX1, vY1, e1a, e2a, hp1 = _build_chain(order=("A", "B"), r1=0.6, r2=0.7, t_shift=0)
    engine1 = DefaultAbstractionEngine(hg1, AbstractionParams(provenance=True))
    eid1 = engine1.promote(hp1)
    hoe1 = hg1.get_edge(EdgeId(str(eid1)))
    assert hoe1 is not None
    label1 = hoe1.attributes.get("provenance", {}).get("hyperpath_label")

    # Promote on a time-shifted copy (+100)
    hg2, vA2, vB2, vX2, vY2, e1b, e2b, hp2 = _build_chain(order=("A", "B"), r1=0.6, r2=0.7, t_shift=100)
    engine2 = DefaultAbstractionEngine(hg2, AbstractionParams(provenance=True))
    eid2 = engine2.promote(hp2)
    hoe2 = hg2.get_edge(EdgeId(str(eid2)))
    assert hoe2 is not None
    label2 = hoe2.attributes.get("provenance", {}).get("hyperpath_label")

    assert isinstance(label1, str) and isinstance(label2, str)
    assert label1 == label2