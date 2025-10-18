# Copyright (c) 2025 DCH Maintainers
# License: MIT

from __future__ import annotations

import math

from dch_core.hypergraph_mem import InMemoryHypergraph
from dch_core.interfaces import Event, Hyperedge, Hyperpath, make_edge_id
from dch_core.crc import CRCExtractor


def test_crc_extractor_basic_card_properties():
    # Build tiny in-memory hypergraph
    hg = InMemoryHypergraph()

    # Vertices: A(100) -> X(150) -> C(200)
    vA = hg.ingest_event(Event(neuron_id=1, t=100))
    vX = hg.ingest_event(Event(neuron_id=2, t=150))
    vC = hg.ingest_event(Event(neuron_id=3, t=200))

    # Edge e1: {A} -> X with informative Beta counters
    tail_e1 = {vA.id}
    e1 = Hyperedge(
        id=make_edge_id(head=vX.id, tail=tail_e1, t=vX.t),
        tail=tail_e1,
        head=vX.id,
        delta_min=vX.t - vA.t,
        delta_max=vX.t - vA.t,
        refractory_rho=0,
        reliability=0.7,
        counts_success=12,
        counts_miss=4,
        provenance="unit:e1",
    )

    # Edge e2: {X} -> C with informative Beta counters
    tail_e2 = {vX.id}
    e2 = Hyperedge(
        id=make_edge_id(head=vC.id, tail=tail_e2, t=vC.t),
        tail=tail_e2,
        head=vC.id,
        delta_min=vC.t - vX.t,
        delta_max=vC.t - vX.t,
        refractory_rho=0,
        reliability=0.5,
        counts_success=8,
        counts_miss=6,
        provenance="unit:e2",
    )

    _ = hg.insert_hyperedges([e1, e2])

    # Construct a trivial Hyperpath using these two edges
    hp = Hyperpath(
        head=vC.id,
        edges=(e1.id, e2.id),
        score=float(e1.reliability) * float(e2.reliability),
        length=2,
        label="A&B->C",
    )

    # Extract CRC with moderate MC samples to keep test fast
    extractor = CRCExtractor(hypergraph=hg, alpha0=1.0, beta0=1.0, ci_level=0.95, samples=2000)
    card = extractor.make_card(label="A&B->C", hyperpath=hp, support=42)

    # Reliability statistics within [0,1] and ordered
    assert 0.0 < card.reliability_mean <= 1.0
    lo, hi = card.reliability_ci
    assert 0.0 <= lo < hi <= 1.0
    assert lo < card.reliability_mean < hi

    # Type classification is one of the expected values
    assert card.type in {"Developing", "Frozen"}

    # Text contains label and CI formatting
    assert "A&B->C" in card.text
    assert "CI_" in card.text
    assert "%" in card.text  # percent formatting present

    # Provenance edges should match hyperpath edges order by string
    expected_edges = [str(e1.id), str(e2.id)]
    assert card.provenance_edges == expected_edges