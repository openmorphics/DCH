# tests/synthetic_micro.py
"""
Synthetic micro-benchmarks for DCH core invariants.
- Temporal window validity (Section 1, 2)
- B-connectivity traversal precondition (Section 5)
- Canonicalization idempotency for FSM templates (Section 6)
These tests are self-contained and do not depend on dch_core yet.
"""
from dataclasses import dataclass
from typing import List, Tuple, Iterable

US_PER_MS = 1000

@dataclass(frozen=True)
class Event:
    neuron_id: int
    t_us: int

@dataclass(frozen=True)
class Hyperedge:
    tail: Tuple[Event, ...]
    head: Event
    delta_min_us: int
    delta_max_us: int

def is_temporally_valid(edge: Hyperedge) -> bool:
    """
    All tail events must precede the head and lie within [delta_min_us, delta_max_us].
    """
    h_t = edge.head.t_us
    for u in edge.tail:
        if not (u.t_us < h_t):
            return False
        delta = h_t - u.t_us
        if not (edge.delta_min_us <= delta <= edge.delta_max_us):
            return False
    return True

def b_connectivity_traversable(edge: Hyperedge, available: Iterable[Event]) -> bool:
    """
    B-connectivity requires that all tail events are present in 'available'.
    """
    avail_set = set(available)
    return all(u in avail_set for u in edge.tail)

# Delay buckets in milliseconds for canonicalization (Section 6 defaults)
DELAY_BUCKETS_MS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]

def bucketize_delay_us(delta_us: int) -> int:
    """
    Map a delay (us) to the smallest bucket (ms) that is >= delay.
    """
    # Convert to ms, ceil-like against bucket thresholds
    ms = (delta_us + US_PER_MS - 1) // US_PER_MS
    for b in DELAY_BUCKETS_MS:
        if ms <= b:
            return b
    return DELAY_BUCKETS_MS[-1]

def canonicalize_hyperpath(edges: List[Hyperedge]) -> str:
    """
    Canonical labeling string for a grounded hyperpath:
    For each edge, token = H:{head_n};M:{m};T:[(i,Δb),...]
    Tail multiset is sorted by (neuron_id, Δb). Edges are sorted by head time, then token.
    """
    def edge_token(e: Hyperedge) -> str:
        h = e.head
        pairs = []
        for u in e.tail:
            delta = h.t_us - u.t_us
            db = bucketize_delay_us(delta)
            pairs.append((u.neuron_id, db))
        pairs.sort()
        m = len(pairs)
        return f"H:{h.neuron_id};M:{m};T:{pairs}"
    # Sort edges by head time then token to stabilize ordering
    sorted_edges = sorted(edges, key=lambda e: (e.head.t_us, edge_token(e)))
    tokens = [edge_token(e) for e in sorted_edges]
    return "|".join(tokens)

# --------------------------
# Tests (runnable via pytest or plain Python)
# --------------------------

def test_temporal_window_validity():
    a = Event(1, 10000)   # 10 ms
    b = Event(2, 11700)   # 11.7 ms
    c = Event(3, 21000)   # 21 ms
    e = Hyperedge(tail=(a, b), head=c, delta_min_us=1000, delta_max_us=30000)
    assert is_temporally_valid(e)

def test_b_connectivity_requires_all_tails():
    a = Event(1, 10000)
    b = Event(2, 11700)
    c = Event(3, 21000)
    e = Hyperedge(tail=(a, b), head=c, delta_min_us=1000, delta_max_us=30000)
    # Only 'a' is available -> should not traverse
    assert not b_connectivity_traversable(e, available=[a])
    # Both are available -> ok
    assert b_connectivity_traversable(e, available=[a, b])

def test_canonicalization_idempotent_permutation():
    a = Event(1, 10000)
    b = Event(2, 11700)
    c = Event(3, 21000)
    e1 = Hyperedge(tail=(a, b), head=c, delta_min_us=1000, delta_max_us=30000)
    e2 = Hyperedge(tail=(b, a), head=c, delta_min_us=1000, delta_max_us=30000)
    s1 = canonicalize_hyperpath([e1])
    s2 = canonicalize_hyperpath([e2])
    assert s1 == s2

if __name__ == "__main__":
    # Simple ad-hoc runner for environments without pytest
    test_temporal_window_validity()
    test_b_connectivity_requires_all_tails()
    test_canonicalization_idempotent_permutation()
    print("Synthetic micro-benchmarks: OK")