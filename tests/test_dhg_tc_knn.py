from __future__ import annotations

from typing import List

from dch_core.interfaces import Event, is_temporally_admissible
from dch_core.hypergraph_mem import InMemoryHypergraph, StaticGraphConnectivity
from dch_core.dhg import DefaultDHGConstructor


def test_tc_knn_generates_unary_and_pair_candidates():
    # In-memory backend and static connectivity:
    # postsyn neuron 10 has presyn sources {1, 2}
    hypergraph = InMemoryHypergraph()
    connectivity = StaticGraphConnectivity({10: [1, 2]})
    dhg = DefaultDHGConstructor()

    # Presynaptic spikes within plausible window relative to head at t=1000
    # Window will be [t_head - delay_max, t_head - delay_min] = [500, 900]
    # Choose presyn spike times within [500, 900]
    presyn_events = [
        Event(neuron_id=1, t=600),
        Event(neuron_id=2, t=700),
        Event(neuron_id=1, t=800),
    ]
    for ev in presyn_events:
        hypergraph.ingest_event(ev)

    # Head (postsyn) spike
    head_t = 1000
    head_ev = Event(neuron_id=10, t=head_t)
    head_vertex = hypergraph.ingest_event(head_ev)

    delay_min = 100
    delay_max = 500
    window = (head_t - delay_max, head_t - delay_min)  # (500, 900)

    # Generate candidates with enough budget to include unary and pair combinations
    candidates = dhg.generate_candidates_tc_knn(
        hypergraph=hypergraph,
        connectivity=connectivity,
        head_vertex=head_vertex,
        window=window,
        k=3,
        combination_order_max=2,        # allow pairs
        causal_coincidence_delta=120,   # cluster [600,700,800] together
        budget_per_head=10,
        init_reliability=0.10,
        refractory_rho=0,
    )

    # Expect: 3 unary (each presyn spike) + 3 pair combinations from the cluster of 3
    # Possible pairs: (600,700), (600,800), (700,800)
    assert len(candidates) == 6, f"Expected 6 candidates, got {len(candidates)}"

    # Validate temporal admissibility and basic attributes
    unary = 0
    pairs = 0
    for e in candidates:
        tail_times = [hypergraph.get_vertex(vid).t for vid in e.tail]
        assert is_temporally_admissible(
            tail_times, head_t, e.delta_min, e.delta_max
        ), "Candidate violates temporal admissibility"
        # Count by tail size
        if len(e.tail) == 1:
            unary += 1
        elif len(e.tail) == 2:
            pairs += 1
        else:
            raise AssertionError("Only unary and pair candidates expected in this test")

    assert unary == 3, f"Expected 3 unary candidates, got {unary}"
    assert pairs == 3, f"Expected 3 pair candidates, got {pairs}"

    # Budgeting should prefer unary (smaller tails) given the scoring heuristic
    candidates_budgeted = dhg.generate_candidates_tc_knn(
        hypergraph=hypergraph,
        connectivity=connectivity,
        head_vertex=head_vertex,
        window=window,
        k=3,
        combination_order_max=2,
        causal_coincidence_delta=120,
        budget_per_head=2,   # enforce a tight budget
        init_reliability=0.10,
        refractory_rho=0,
    )
    assert len(candidates_budgeted) == 2, "Budget should reduce candidate count to 2"
    assert all(
        len(e.tail) == 1 for e in candidates_budgeted
    ), "Budgeting should prefer unary candidates under the scoring heuristic"