# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Beta utilities (stdlib-only) for Beta–Bernoulli posterior computation and Monte Carlo
uncertainty quantification.

This module provides:
- posterior_params(edge, alpha0, beta0): add Hyperedge counters to Beta prior
- posterior_mean(alpha_post, beta_post): mean of Beta posterior
- sample_beta(alpha_post, beta_post, n, seed): Monte Carlo sampling via random.betavariate
- credible_interval_mc(alpha_post, beta_post, level, n, seed): MC credible interval by quantiles

References and reuse:
- Hyperedge counters: counts_success, counts_miss (see dch_core.interfaces.Hyperedge)
- Priors consistent with BetaPlasticityEngine (see dch_core.plasticity_beta.BetaPlasticityEngine)

Determinism:
- When a seed is provided, sampling uses random.Random(seed).betavariate to ensure
  deterministic sequences across runs (sufficient for P1).
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Optional

from dch_core.interfaces import Hyperedge


def posterior_params(edge: Hyperedge, alpha0: float, beta0: float) -> Tuple[float, float]:
    """
    Compute Beta–Bernoulli posterior parameters from a Hyperedge and Beta prior.

    Given:
        - Prior Beta(alpha0, beta0), with alpha0 > 0, beta0 > 0
        - Edge counters:
            * counts_success: number of Bernoulli successes observed for this edge
            * counts_miss: number of Bernoulli failures/misses observed for this edge

    Returns:
        (alpha_post, beta_post) where:
            alpha_post = alpha0 + edge.counts_success
            beta_post  = beta0  + edge.counts_miss

    Notes:
        - This utility reuses the existing counters stored on Hyperedge (no protocol changes).
        - The returned parameters are floats to keep a consistent numerical API, even though
          counters are integers.
    """
    if alpha0 <= 0.0 or beta0 <= 0.0:
        raise ValueError("alpha0 and beta0 must be positive")
    alpha_post = float(alpha0) + float(edge.counts_success)
    beta_post = float(beta0) + float(edge.counts_miss)
    return alpha_post, beta_post


def posterior_mean(alpha_post: float, beta_post: float) -> float:
    """
    Compute the posterior mean for a Beta(alpha_post, beta_post) distribution.

    Formula:
        E[p | data] = alpha_post / (alpha_post + beta_post)

    Args:
        alpha_post: posterior alpha (must be > 0)
        beta_post: posterior beta (must be > 0)

    Returns:
        Posterior mean in [0, 1].

    Numerical properties:
        - The result is exactly alpha_post / (alpha_post + beta_post).
        - Assumes positive parameters; raises ValueError otherwise.
    """
    if alpha_post <= 0.0 or beta_post <= 0.0:
        raise ValueError("alpha_post and beta_post must be positive")
    return float(alpha_post) / float(alpha_post + beta_post)


def sample_beta(
    alpha_post: float,
    beta_post: float,
    n: int = 10000,
    seed: Optional[int] = None,
) -> List[float]:
    """
    Draw Monte Carlo samples from Beta(alpha_post, beta_post) using stdlib only.

    Sampling:
        - Uses Python's random.betavariate (no torch, no external deps).
        - If seed is provided, constructs random.Random(seed) and samples from that
          instance for deterministic sequences.

    Args:
        alpha_post: posterior alpha (> 0)
        beta_post: posterior beta (> 0)
        n: number of samples (n >= 1)
        seed: optional integer seed for determinism

    Returns:
        List of length n containing samples in [0, 1].
    """
    if alpha_post <= 0.0 or beta_post <= 0.0:
        raise ValueError("alpha_post and beta_post must be positive")
    if n <= 0:
        raise ValueError("n must be positive")

    rng = random.Random(seed) if seed is not None else random
    return [rng.betavariate(alpha_post, beta_post) for _ in range(int(n))]


def _quantile(sorted_x: List[float], p: float) -> float:
    """
    Compute a linear-interpolated quantile for 0 <= p <= 1 on a pre-sorted list.

    Method:
        - Uses the 'linear' method:
            pos = p * (N - 1)
            i = floor(pos), j = ceil(pos)
            q = x[i] if i == j else x[i] + (pos - i) * (x[j] - x[i])

    Args:
        sorted_x: sorted list of floats (non-empty)
        p: quantile probability in [0, 1]

    Returns:
        The p-quantile value.
    """
    if not sorted_x:
        raise ValueError("sorted_x must be non-empty")
    if p < 0.0 or p > 1.0:
        raise ValueError("p must be in [0, 1]")

    n = len(sorted_x)
    if n == 1:
        return float(sorted_x[0])

    pos = p * (n - 1)
    i = int(math.floor(pos))
    j = int(math.ceil(pos))
    if i == j:
        return float(sorted_x[i])
    lo = float(sorted_x[i])
    hi = float(sorted_x[j])
    w = pos - i
    return float(lo + w * (hi - lo))


def credible_interval_mc(
    alpha_post: float,
    beta_post: float,
    level: float = 0.95,
    n: int = 20000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Monte Carlo credible interval for Beta(alpha_post, beta_post) via sampling.

    Approach:
        - Draw n samples using sample_beta (stdlib random.betavariate).
        - Compute lower/upper quantiles at:
            p_lo = (1 - level) / 2
            p_hi = 1 - p_lo
        - Return (lo, hi) with 0 <= lo < hi <= 1.

    Determinism:
        - If seed is provided, the CI is deterministic (reproducible) for the given
          (alpha_post, beta_post, level, n, seed), as sampling uses random.Random(seed).

    Notes:
        - This is an approximation (Monte Carlo); sufficient for P1 acceptance tests.
        - For exact quantiles one could use the inverse incomplete beta, but that would
          require non-stdlib dependencies.

    Args:
        alpha_post: posterior alpha (> 0)
        beta_post: posterior beta (> 0)
        level: mass contained in the interval (e.g., 0.95)
        n: number of MC samples (recommend 10k–20k for stability)
        seed: optional seed for deterministic sampling

    Returns:
        (lo, hi) tuple representing the (approximate) credible interval.
    """
    if alpha_post <= 0.0 or beta_post <= 0.0:
        raise ValueError("alpha_post and beta_post must be positive")
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0, 1)")
    if n <= 0:
        raise ValueError("n must be positive")

    samples = sample_beta(alpha_post, beta_post, n=n, seed=seed)
    xs = sorted(samples)
    tail = (1.0 - float(level)) / 2.0
    p_lo = max(0.0, min(1.0, tail))
    p_hi = max(0.0, min(1.0, 1.0 - tail))
    lo = _quantile(xs, p_lo)
    hi = _quantile(xs, p_hi)
    # Enforce monotonic and bounds by construction; slight numeric jitter safeguard:
    lo = max(0.0, min(1.0, lo))
    hi = max(0.0, min(1.0, hi))
    if not (lo <= hi):
        # Extremely unlikely with linear quantiles; fallback to min/max
        lo, hi = float(xs[0]), float(xs[-1])
    return float(lo), float(hi)


__all__ = [
    "posterior_params",
    "posterior_mean",
    "sample_beta",
    "credible_interval_mc",
]