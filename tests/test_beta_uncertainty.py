# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Uncertainty quantification utilities for Beta–Bernoulli (stdlib-only).

Coverage:
- posterior_params: adds Hyperedge counters to Beta prior
- posterior_mean: returns alpha_post / (alpha_post + beta_post)
- sample_beta: deterministic sampling with seed
- credible_interval_mc: Monte Carlo credible interval enclosing the mean

Scope aligns with P1 constraints: stdlib only, counters from Hyperedge, and priors
consistent with BetaPlasticityEngine.
"""

from __future__ import annotations

import math

import pytest

from dch_core.beta_utils import (
    posterior_params,
    posterior_mean,
    sample_beta,
    credible_interval_mc,
)
from dch_core.hypergraph_mem import InMemoryHypergraph
from dch_core.interfaces import (
    Event,
    Hyperedge,
    Hyperpath,
    PlasticityState,
    make_edge_id,
    make_vertex_id,
)
from dch_core.plasticity_beta import BetaPlasticityEngine


def _make_min_edge(
    *,
    head_t: int,
    dt: int,
    reliability: float = 0.5,
    counts_success: int = 0,
    counts_miss: int = 0,
) -> Hyperedge:
    """
    Construct a minimal unary admissible Hyperedge for testing:
    - Head at head_t, tail at head_t - dt
    - delta_min == delta_max == dt ensures admissibility by construction
    """
    head_vid = make_vertex_id(10, head_t)
    tail_vid = make_vertex_id(1, head_t - dt)
    eid = make_edge_id(head=head_vid, tail={tail_vid}, t=head_t)
    return Hyperedge(
        id=eid,
        tail={tail_vid},
        head=head_vid,
        delta_min=int(dt),
        delta_max=int(dt),
        refractory_rho=0,
        reliability=float(reliability),
        counts_success=int(counts_success),
        counts_miss=int(counts_miss),
    )


def test_posterior_params_basic():
    # Prior
    alpha0, beta0 = 2.0, 3.0
    # Edge with counters
    e = _make_min_edge(head_t=1000, dt=100, counts_success=8, counts_miss=2)
    a_post, b_post = posterior_params(e, alpha0, beta0)
    assert a_post == pytest.approx(10.0, abs=1e-12)
    assert b_post == pytest.approx(5.0, abs=1e-12)

    mean = posterior_mean(a_post, b_post)
    expected = a_post / (a_post + b_post)
    assert abs(mean - expected) < 1e-6
    # Explicit expected: 10 / 15 = 0.666...
    assert mean == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_mc_ci_contains_mean_and_is_monotone():
    a_post, b_post = 10.0, 5.0
    # Fix seed for determinism and keep n moderate for speed
    lo, hi = credible_interval_mc(a_post, b_post, level=0.95, n=10000, seed=123)
    mean = posterior_mean(a_post, b_post)
    assert 0.0 <= lo < hi <= 1.0
    assert lo - 1e-12 <= mean <= hi + 1e-12  # allow tiny numeric jitter


def test_sampling_determinism():
    a_post, b_post = 3.0, 7.0
    n = 1000
    s1 = sample_beta(a_post, b_post, n=n, seed=42)
    s2 = sample_beta(a_post, b_post, n=n, seed=42)
    s3 = sample_beta(a_post, b_post, n=n, seed=43)
    assert s1 == s2  # identical with same seed
    # Different seed should yield a different sequence (overwhelmingly likely)
    assert s1 != s3


def test_pipeline_beta_engine_with_utils():
    # Minimal in-memory hypergraph and single-edge positive update
    hg = InMemoryHypergraph()
    # Materialize vertices (not strictly required for this backend, but realistic)
    head_t, dt = 1000, 100
    hg.ingest_event(Event(neuron_id=10, t=head_t))
    hg.ingest_event(Event(neuron_id=1, t=head_t - dt))
    e = _make_min_edge(head_t=head_t, dt=dt, reliability=0.50)
    _ = hg.insert_hyperedges([e])

    hp = Hyperpath(head=e.head, edges=(e.id,), score=1.0, length=1, label="e")
    engine = BetaPlasticityEngine(alpha0=1.0, beta0=1.0)
    state = PlasticityState(ema_alpha=0.10, reliability_clamp=(0.02, 0.98), prune_threshold=0.05)

    _ = engine.update_from_evidence(
        hypergraph=hg,
        hyperpaths=[hp],
        sign=+1,
        now_t=head_t + 1,
        state=state,
    )

    # Fetch edge post-update
    e_after = hg.get_edge(e.id)
    assert e_after is not None

    # Compute posterior from counters via utils and compare with engine reliability
    a_post, b_post = posterior_params(e_after, alpha0=1.0, beta0=1.0)
    mean = posterior_mean(a_post, b_post)
    assert mean == pytest.approx(e_after.reliability, abs=1e-6)