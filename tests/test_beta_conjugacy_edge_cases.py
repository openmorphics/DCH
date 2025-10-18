from __future__ import annotations

import math

from dch_core.interfaces import Hyperedge, PlasticityState, make_vertex_id
from dch_core.beta_utils import posterior_params, posterior_mean
from dch_core.plasticity_beta import BetaPlasticityEngine
from dch_core.hypergraph_mem import InMemoryHypergraph
from dch_core.interfaces import Hyperpath


def _make_edge_zero_counts() -> Hyperedge:
    head = make_vertex_id(10, 1000)
    tail = {make_vertex_id(1, 600)}
    return Hyperedge(
        id=f"{head}&{sorted(list(tail))[0]}#seed",
        tail=tail,
        head=head,
        delta_min=100,
        delta_max=500,
        refractory_rho=0,
        reliability=0.5,  # arbitrary; not used by posterior utilities
        counts_success=0,
        counts_miss=0,
    )


def test_extreme_priors_zero_counts():
    # Prior (alpha0, beta0) very small but positive
    alpha0, beta0 = 1e-3, 1e-3
    e = _make_edge_zero_counts()
    a_post, b_post = posterior_params(e, alpha0, beta0)
    m = posterior_mean(a_post, b_post)
    assert 0.0 < m < 1.0
    # Should be close to 0.5 for symmetric tiny prior
    assert abs(m - 0.5) <= 0.05


def test_large_counts_stability():
    # Large counts with modest prior should yield stable mean near empirical rate
    alpha0, beta0 = 2.0, 3.0
    head = make_vertex_id(10, 1000)
    tail = {make_vertex_id(1, 600)}
    e = Hyperedge(
        id=f"{head}&{sorted(list(tail))[0]}#large",
        tail=tail,
        head=head,
        delta_min=100,
        delta_max=500,
        refractory_rho=0,
        reliability=0.5,
        counts_success=10000,
        counts_miss=5000,
    )
    a_post, b_post = posterior_params(e, alpha0, beta0)
    m = posterior_mean(a_post, b_post)
    expected = (alpha0 + 10000.0) / (alpha0 + beta0 + 10000.0 + 5000.0)
    assert math.isclose(m, expected, rel_tol=0.0, abs_tol=1e-6)


def test_negative_evidence_effect_with_engine():
    # Build minimal hypergraph with a single edge and apply negative update
    hg = InMemoryHypergraph()
    head = make_vertex_id(10, 1000)
    tail_vid = make_vertex_id(1, 600)
    e = Hyperedge(
        id=f"{head}&{tail_vid}#neg",
        tail={tail_vid},
        head=head,
        delta_min=100,
        delta_max=500,
        refractory_rho=0,
        reliability=0.9,  # start high so a negative update should decrease it
        counts_success=0,
        counts_miss=0,
    )
    hg.insert_hyperedges([e])

    # Prepare a synthetic hyperpath that references the edge
    hp = Hyperpath(head=head, edges=(e.id,), score=1.0, length=1, label=None)

    eng = BetaPlasticityEngine(alpha0=1.0, beta0=1.0)
    state = PlasticityState()  # default clamp [0.02, 0.98]
    r_before = e.reliability
    updates = eng.update_from_evidence(hypergraph=hg, hyperpaths=[hp], sign=-1, now_t=1000, state=state)

    assert e.id in updates
    r_after = updates[e.id]
    # Reliability should decrease under negative evidence
    assert r_after < r_before
    # Counters should reflect a negative "miss"
    assert e.counts_miss >= 1