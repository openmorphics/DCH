from __future__ import annotations

from typing import List, Tuple

from dch_core.interfaces import (
    Event,
    EdgeId,
    Hyperedge,
    Hyperpath,
    PlasticityState,
)
from dch_core.hypergraph_mem import InMemoryHypergraph
from dch_core.traversal import DefaultTraversalEngine
from dch_core.plasticity import DefaultPlasticityEngine


def _build_small_hypergraph() -> Tuple[InMemoryHypergraph, EdgeId, EdgeId, EdgeId, str]:
    """
    Construct a tiny hypergraph for testing traversal and plasticity:

    Presyn spikes:
      - n=1 at t=800 (v1)
      - n=2 at t=900 (v2)
    Head spike:
      - n=10 at t=1000 (vh)

    Insert edges into the hypergraph:
      - e1: tail={v2}, head=vh, reliability=0.8
      - e2: tail={v1}, head=vh, reliability=0.5
      - e3: tail={v1,v2}, head=vh, reliability=0.7
    """
    hg = InMemoryHypergraph()

    # Ingest vertices
    v1 = hg.ingest_event(Event(neuron_id=1, t=800))
    v2 = hg.ingest_event(Event(neuron_id=2, t=900))
    vh = hg.ingest_event(Event(neuron_id=10, t=1000))

    # Edge temporal window: each tail time ∈ [head - delta_max, head - delta_min]
    delta_min = 50
    delta_max = 250
    refractory_rho = 0

    # Define edges (EdgeIds are arbitrary unique strings for this test)
    e1 = Hyperedge(
        id=EdgeId("e1"),
        tail={v2.id},
        head=vh.id,
        delta_min=delta_min,
        delta_max=delta_max,
        refractory_rho=refractory_rho,
        reliability=0.8,
        provenance="test",
    )
    e2 = Hyperedge(
        id=EdgeId("e2"),
        tail={v1.id},
        head=vh.id,
        delta_min=delta_min,
        delta_max=delta_max,
        refractory_rho=refractory_rho,
        reliability=0.5,
        provenance="test",
    )
    e3 = Hyperedge(
        id=EdgeId("e3"),
        tail={v1.id, v2.id},
        head=vh.id,
        delta_min=delta_min,
        delta_max=delta_max,
        refractory_rho=refractory_rho,
        reliability=0.7,
        provenance="test",
    )

    # Insert edges
    admitted = hg.insert_hyperedges([e1, e2, e3])
    assert set(admitted) == {EdgeId("e1"), EdgeId("e2"), EdgeId("e3")}

    return hg, e1.id, e2.id, e3.id, vh.id  # (hypergraph, e1, e2, e3, head_vid)


def test_traversal_basic_backward_and_scoring():
    hg, e1, e2, e3, head_vid = _build_small_hypergraph()
    vh = hg.get_vertex(head_vid)
    assert vh is not None

    # Run traversal
    trav = DefaultTraversalEngine()
    hyperpaths = trav.backward_traverse(
        hypergraph=hg,
        target=vh,
        horizon=1000,
        beam_size=8,
        rng=None,
        refractory_enforce=True,
    )

    # Expect three single-edge hyperpaths corresponding to the three incoming edges to head
    labels = set(hp.label for hp in hyperpaths if hp.label is not None)
    assert {"e1", "e2", "e3"}.issubset(labels), f"Expected labels for e1,e2,e3 in results, got {labels}"

    # Best-scoring path should correspond to the highest-reliability edge (e1: 0.8)
    best = max(hyperpaths, key=lambda hp: hp.score)
    assert best.label == "e1", f"Expected best path label 'e1', got {best.label}"


def test_plasticity_update_and_prune():
    hg, e1, e2, e3, head_vid = _build_small_hypergraph()

    # Compose simple hyperpaths as evidence:
    # - hp1 uses e1 with higher score weight
    # - hp2 uses e3 with lower score weight
    hp1 = Hyperpath(head=head_vid, edges=(e1,), score=0.9, length=1, label="e1")
    hp2 = Hyperpath(head=head_vid, edges=(e3,), score=0.3, length=1, label="e3")

    plast = DefaultPlasticityEngine()
    pstate = PlasticityState(
        ema_alpha=0.10,
        reliability_clamp=(0.02, 0.98),
        decay_lambda=0.0,
        freeze=False,
        prune_threshold=0.05,
    )

    # Positive update (potentiation)
    updates_pos = plast.update_from_evidence(
        hypergraph=hg,
        hyperpaths=[hp1, hp2],
        sign=+1,
        now_t=1000,
        state=pstate,
    )
    r1_after_pos = hg.get_edge(e1).reliability  # expected to increase above 0.8
    r3_after_pos = hg.get_edge(e3).reliability
    assert r1_after_pos > 0.8, f"Reliability of e1 should increase, got {r1_after_pos}"
    assert r3_after_pos > 0.7, f"Reliability of e3 should increase, got {r3_after_pos}"
    assert hg.get_edge(e1).counts_success == 1
    assert hg.get_edge(e3).counts_success == 1

    # Negative update (depression) on e1 via a single-path evidence
    hp1_neg = Hyperpath(head=head_vid, edges=(e1,), score=1.0, length=1, label="e1")
    updates_neg = plast.update_from_evidence(
        hypergraph=hg,
        hyperpaths=[hp1_neg],
        sign=-1,
        now_t=1001,
        state=pstate,
    )
    r1_after_neg = hg.get_edge(e1).reliability
    assert r1_after_neg < r1_after_pos, "Reliability of e1 should decrease after negative evidence"
    assert hg.get_edge(e1).counts_miss == 1

    # Prune: set e2 low and ensure it gets removed under threshold
    hg.get_edge(e2).reliability = 0.04
    pruned = plast.prune(hg, now_t=1002, state=pstate)
    assert pruned >= 1, "At least one edge (e2) should be pruned"
    assert hg.get_edge(e2) is None, "Edge e2 should be removed by pruning"