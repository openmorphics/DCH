from __future__ import annotations

import math
from typing import List, Tuple

from dch_core.interfaces import Event, Hyperpath, EdgeId, VertexId
from dch_core.embeddings.wl import WLHyperpathEmbedding, WLParams
from dch_data.transforms import (
    time_window,
    neuron_filter,
    time_normalize,
    sort_events,
    subsample,
    chunk_by_bins,
    sliding_windows,
)


def _l2_norm(xs: List[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in xs)) if xs else 0.0


def test_wl_embedding_shape_and_norm_and_determinism():
    # Construct a simple hyperpath label with two edges
    hp = Hyperpath(
        head=VertexId("H"),
        edges=(EdgeId("e1"), EdgeId("e2")),
        score=1.0,
        length=2,
        label="e1|e2",
    )
    params = WLParams(d=32, iters=1, salt=123)
    emb_engine = WLHyperpathEmbedding(params)

    v1 = list(emb_engine.embed(hp))
    assert len(v1) == 32, f"Expected embedding dimension 32, got {len(v1)}"
    n1 = _l2_norm(v1)
    assert abs(n1 - 1.0) < 1e-6, f"Embedding must be L2-normalized, got norm {n1}"

    # Deterministic across calls
    v2 = list(emb_engine.embed(hp))
    assert v1 == v2, "Embedding should be deterministic for the same hyperpath"

    # Order invariance of edges (canonical sort inside the embed)
    hp_shuffled = Hyperpath(
        head=hp.head,
        edges=(EdgeId("e2"), EdgeId("e1")),
        score=hp.score,
        length=hp.length,
        label="e2|e1",
    )
    v3 = list(emb_engine.embed(hp_shuffled))
    assert v1 == v3, "Embedding should be invariant to edge order after canonicalization"


def test_transforms_time_window_and_neuron_filter():
    events = [
        Event(neuron_id=1, t=100),
        Event(neuron_id=2, t=150),
        Event(neuron_id=1, t=200),
        Event(neuron_id=3, t=350),
    ]

    # Time window [120, 300] selects t=150 and t=200
    tw = time_window(events, window=(120, 300))
    assert [e.t for e in tw] == [150, 200], f"Unexpected time_window result: {[e.t for e in tw]}"

    # Neuron allowlist {1,3} then denylist {3} => only neuron 1
    nf_allow = neuron_filter(events, allowlist={1, 3})
    assert set(e.neuron_id for e in nf_allow) <= {1, 3}
    nf_allow_deny = neuron_filter(nf_allow, denylist={3})
    assert all(e.neuron_id == 1 for e in nf_allow_deny)


def test_transforms_time_normalize_and_sort_and_subsample():
    events = [
        Event(neuron_id=2, t=300),
        Event(neuron_id=1, t=100),
        Event(neuron_id=3, t=250),
        Event(neuron_id=1, t=200),
    ]

    # Normalize times so min becomes 0
    normed = time_normalize(events)
    times = [e.t for e in normed]
    assert min(times) == 0, f"Expected min timestamp 0 after normalization, got {min(times)}"

    # Sort events by (t, neuron_id)
    sorted_events = sort_events(events)
    assert [(e.t, e.neuron_id) for e in sorted_events] == [(100, 1), (200, 1), (250, 3), (300, 2)]

    # Subsample stride=2 drops every other event
    sub = subsample(sorted_events, stride=2)
    assert len(sub) == 2 and sub[0].t == 100 and sub[1].t == 250, "Subsample stride=2 failed"


def test_transforms_chunk_by_bins_and_sliding_windows():
    events = [
        Event(neuron_id=1, t=100),
        Event(neuron_id=2, t=150),
        Event(neuron_id=1, t=200),
        Event(neuron_id=3, t=350),
        Event(neuron_id=2, t=420),
    ]
    # Chunk by bins within [100, 349] at width 100: bins cover
    # bin 0: [100..199] -> events at 100,150
    # bin 1: [200..299] -> event at 200
    # bin 2: [300..399] -> none (since 350 is outside window end 349)
    buckets = chunk_by_bins(events, window=(100, 349), bin_width=100)
    assert set(buckets.keys()) == {0, 1}, f"Expected bins 0 and 1, got {set(buckets.keys())}"
    assert [e.t for e in buckets[0]] == [100, 150]
    assert [e.t for e in buckets[1]] == [200]

    # Sliding windows over [100, 420] with size=150 and step=100
    # Windows:
    # [100,250], [200,350], [300,420]
    wins = list(sliding_windows(events, window_size=150, step=100, t_start=100, t_end=420))
    assert len(wins) >= 3, f"Expected at least 3 windows, got {len(wins)}"
    # Check first window content
    (w0, ev0) = wins[0]
    assert w0 == (100, 250)
    assert [e.t for e in ev0] == [100, 150, 200]