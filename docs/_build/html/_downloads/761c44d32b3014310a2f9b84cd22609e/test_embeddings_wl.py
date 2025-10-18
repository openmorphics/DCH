from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from dch_core.interfaces import EdgeId, Hyperedge, Hyperpath, VertexId
from dch_core.embeddings import WLHyperpathEmbedding, WLParams


def mk_vid(n: int, t: int) -> VertexId:
    return VertexId(f"{n}@{t}")


def mk_eid(head: VertexId, tails: Sequence[VertexId], nonce: int = 0) -> EdgeId:
    # Do not sort tails here to allow testing tail-order invariance
    tails_s = ",".join(str(tv) for tv in tails)
    return EdgeId(f"{head}&{tails_s}#{nonce}")


def compose_canonical_eid(head: VertexId, tails: Sequence[VertexId]) -> EdgeId:
    # Compose canonical-like id used by the embedding module for reliability lookup
    tails_s = ",".join(sorted(str(tv) for tv in tails))
    return EdgeId(f"{head}&{tails_s}#0")


def l2(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = l2(a)
    nb = l2(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_edge(head: VertexId, tails: Sequence[VertexId], reliability: float = 1.0) -> Hyperedge:
    return Hyperedge(
        id=mk_eid(head, tails),
        tail=set(tails),
        head=head,
        delta_min=0,
        delta_max=10_000_000,
        refractory_rho=0,
        reliability=float(reliability),
    )


def build_lookup(edges: Iterable[Hyperedge]) -> Dict[EdgeId, Hyperedge]:
    """
    Build an id -> edge lookup that includes both the raw eid and the composed canonical eid
    so that include_reliability=True affects both canonical_label and embedding.
    """
    table: Dict[EdgeId, Hyperedge] = {}
    for e in edges:
        table[e.id] = e
        # Add composed canonical id variant as used internally by WL embedder
        raw_head = VertexId(str(e.head))
        raw_tails = sorted((VertexId(str(tv)) for tv in e.tail), key=str)
        table[compose_canonical_eid(raw_head, raw_tails)] = e
    return table


def test_determinism():
    # Build a hyperpath with two edges to the same sink
    h = mk_vid(3, 1000)
    t1 = mk_vid(1, 980)   # dt = 20
    t2 = mk_vid(2, 965)   # dt = 35
    e1 = mk_eid(h, [t1, t2], nonce=42)

    t3 = mk_vid(2, 940)   # different chain edge
    e2 = mk_eid(t2, [t3], nonce=7)

    hp = Hyperpath(head=h, edges=(e1, e2), score=1.0, length=2, label=None)

    params = WLParams(vector_dim=64, iterations=2, normalize=True, hash_seed=123, time_resolution=1.0)
    emb = WLHyperpathEmbedding(params)

    v1 = emb.embed(hp)
    v2 = emb.embed(hp)
    assert np.array_equal(v1, v2), "Embedding must be deterministic across calls"

    # Canonical label deterministic
    c1 = emb.canonical_label(hp)
    c2 = emb.canonical_label(hp)
    assert c1 == c2, "Canonical label must be deterministic"


def test_tail_permutation_invariance():
    h = mk_vid(5, 2000)
    a = mk_vid(1, 1970)  # dt=30
    b = mk_vid(2, 1985)  # dt=15

    e_orig = mk_eid(h, [a, b], nonce=0)
    e_perm = mk_eid(h, [b, a], nonce=0)  # different tail order in the string

    hp1 = Hyperpath(head=h, edges=(e_orig,), score=1.0, length=1, label=None)
    hp2 = Hyperpath(head=h, edges=(e_perm,), score=1.0, length=1, label=None)

    emb = WLHyperpathEmbedding(WLParams(vector_dim=64, iterations=2, normalize=True))

    v1 = emb.embed(hp1)
    v2 = emb.embed(hp2)
    assert np.array_equal(v1, v2), "Embedding should be invariant to tail permutation within an edge"

    c1 = emb.canonical_label(hp1)
    c2 = emb.canonical_label(hp2)
    assert c1 == c2, "Canonical label should be invariant to tail permutation within an edge"


def test_time_shift_invariance():
    # Original
    offset = 12345
    h = mk_vid(9, 10_000)
    a = mk_vid(4, 9_960)   # dt=40
    b = mk_vid(7, 9_950)   # dt=50
    e = mk_eid(h, [a, b], nonce=11)
    hp = Hyperpath(head=h, edges=(e,), score=1.0, length=1, label=None)

    # Shift all timestamps by a constant
    h_s = mk_vid(9, 10_000 + offset)
    a_s = mk_vid(4, 9_960 + offset)
    b_s = mk_vid(7, 9_950 + offset)
    e_s = mk_eid(h_s, [a_s, b_s], nonce=11)
    hp_s = Hyperpath(head=h_s, edges=(e_s,), score=1.0, length=1, label=None)

    emb = WLHyperpathEmbedding(WLParams(vector_dim=64, iterations=2, normalize=True, time_resolution=1.0))
    v = emb.embed(hp)
    v_shift = emb.embed(hp_s)
    assert np.array_equal(v, v_shift), "Embedding should be invariant to absolute time shifts (Δt only)"

    c = emb.canonical_label(hp)
    c_shift = emb.canonical_label(hp_s)
    assert c == c_shift, "Canonical label should be invariant to absolute time shifts (Δt only)"


def test_discrimination():
    h = mk_vid(10, 5000)
    # Hyperpath 1
    a1 = mk_vid(1, 4980)  # dt=20
    b1 = mk_vid(2, 4970)  # dt=30
    e1 = mk_eid(h, [a1, b1], nonce=0)
    hp1 = Hyperpath(head=h, edges=(e1,), score=1.0, length=1, label=None)

    # Hyperpath 2 (different delays)
    a2 = mk_vid(1, 4990)  # dt=10 (changed)
    b2 = mk_vid(2, 4970)  # dt=30
    e2 = mk_eid(h, [a2, b2], nonce=0)
    hp2 = Hyperpath(head=h, edges=(e2,), score=1.0, length=1, label=None)

    emb = WLHyperpathEmbedding(WLParams(vector_dim=128, iterations=2, normalize=True))
    v1 = emb.embed(hp1)
    v2 = emb.embed(hp2)

    # Both are normalized; cosine should be < 1
    cos = cosine(v1, v2)
    assert cos < 0.999999, f"Structurally different hyperpaths should not have identical embeddings (cos={cos})"
    assert not np.allclose(v1, v2), "Embeddings should differ numerically for different structures"


def test_normalization_and_zero_vector():
    h = mk_vid(3, 3000)
    a = mk_vid(1, 2980)
    e = mk_eid(h, [a], nonce=0)
    hp_nonempty = Hyperpath(head=h, edges=(e,), score=1.0, length=1, label=None)

    emb = WLHyperpathEmbedding(WLParams(vector_dim=32, iterations=1, normalize=True))
    v = emb.embed(hp_nonempty)
    n = l2(v)
    assert abs(n - 1.0) < 1e-9, f"Non-empty embedding should be L2-normalized, got norm={n}"

    # Empty hyperpath -> zero vector
    hp_empty = Hyperpath(head=h, edges=tuple(), score=1.0, length=0, label=None)
    z = emb.embed(hp_empty)
    assert np.allclose(z, 0.0), "Empty hyperpath should map to the zero vector"
    assert l2(z) == 0.0


def test_include_reliability_effect():
    h = mk_vid(8, 8000)
    a = mk_vid(1, 7980)
    b = mk_vid(2, 7970)
    tails = [a, b]
    eid = mk_eid(h, tails, nonce=99)
    hp = Hyperpath(head=h, edges=(eid,), score=1.0, length=1, label=None)

    # Build two different reliability lookups
    e_low = build_edge(h, tails, reliability=0.2)
    e_high = build_edge(h, tails, reliability=0.9)

    lookup_low = build_lookup([e_low])
    lookup_high = build_lookup([e_high])

    params_rel = WLParams(vector_dim=64, iterations=2, normalize=True, include_reliability=True)
    emb_low = WLHyperpathEmbedding(params_rel, edge_lookup=lookup_low)
    emb_high = WLHyperpathEmbedding(params_rel, edge_lookup=lookup_high)

    v_low = emb_low.embed(hp)
    v_high = emb_high.embed(hp)
    assert not np.allclose(v_low, v_high), "Different reliabilities should produce different embeddings when include_reliability=True"

    c_low = emb_low.canonical_label(hp)
    c_high = emb_high.canonical_label(hp)
    assert c_low != c_high, "Canonical label should include reliability when include_reliability=True"

    # With include_reliability=False, embeddings should match regardless of reliability table
    params_no_rel = WLParams(vector_dim=64, iterations=2, normalize=True, include_reliability=False)
    emb_no_rel_low = WLHyperpathEmbedding(params_no_rel, edge_lookup=lookup_low)
    emb_no_rel_high = WLHyperpathEmbedding(params_no_rel, edge_lookup=lookup_high)
    v_nr_low = emb_no_rel_low.embed(hp)
    v_nr_high = emb_no_rel_high.embed(hp)
    assert np.array_equal(v_nr_low, v_nr_high), "Embeddings should be identical when include_reliability=False"