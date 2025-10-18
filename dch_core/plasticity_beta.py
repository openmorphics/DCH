# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Bayesian Beta–Bernoulli plasticity engine for Dynamic Causal Hypergraph (DCH).

Implements a PlasticityEngine that treats edge reliability as the posterior mean of a
Beta(alpha0, beta0) prior updated by evidence from traversed Hyperpaths.

Key mechanics
- Evidence aggregation matches the EMA engine:
  * For each Hyperpath hp with score s=hp.score>=0, add s to every edge on that path.
  * Normalize per-edge contribution by the total sum over all hyperpaths.
    This yields s_norm(e) in [0,1] with sum over updated edges == 1.
- Beta–Bernoulli updates:
  * Posterior parameters (conceptually) accumulate per-edge as:
      successes += s_norm(e)     if sign > 0
      misses    += share_negative * s_norm(e)  if sign < 0
  * Posterior mean reliability:
      rho(e) = (alpha0 + successes) / (alpha0 + beta0 + successes + misses)
- Clamping:
  * The resulting reliability is clamped via PlasticityState.clamp() to keep values
    within configured bounds.
- Counters:
  * Hyperedge.counts_success / counts_miss are stored as integers in the data model.
    We use fractional evidence only for computing the posterior mean reliability.
    For counters, we add int(round(contribution)) so single-edge evidence increments
    by exactly 1 while preserving integer counters.

References:
- Protocols and data models: dch_core.interfaces (see PlasticityEngine)
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

from dch_core.interfaces import (
    EdgeId,
    Hyperedge,
    HypergraphOps,
    Hyperpath,
    PlasticityEngine,
    PlasticityState,
    ReliabilityScore,
    Timestamp,
)

from dch_core.beta_utils import posterior_params as _betautils_posterior_params


def edge_posterior_params(edge: Hyperedge, alpha0: float, beta0: float) -> tuple[float, float]:
    """
    Thin wrapper for dch_core.beta_utils.posterior_params. Provided for discoverability
    alongside the Beta engine without modifying any protocols.
    """
    return _betautils_posterior_params(edge, alpha0, beta0)


class BetaPlasticityEngine(PlasticityEngine):
    """
    Beta–Bernoulli plasticity engine.

    Parameters
    - alpha0: float = 1.0
        Beta prior alpha (pseudo-successes)
    - beta0: float = 1.0
        Beta prior beta (pseudo-failures)
    - share_negative: float = 1.0
        Scale for negative evidence contribution (applied to misses)
    - respect_freeze: bool = True
        If True and state.freeze is set, updates are skipped (no-op).

    Posterior mean update per edge e:
        rho(e) = (alpha0 + s_e) / (alpha0 + beta0 + s_e + f_e)
    where s_e and f_e are the cumulative (conceptual) successes and failures after
    adding the current normalized evidence contribution.

    Notes on counters:
    - The Hyperedge model stores counts_success/counts_miss as ints. We keep them
      integer by adding int(round(contribution)). Fractional evidence is reflected
      in reliability only, not in the stored counters, per P1 guidance.
    """

    def __init__(self, alpha0: float = 1.0, beta0: float = 1.0, share_negative: float = 1.0, respect_freeze: bool = True) -> None:
        if alpha0 <= 0.0 or beta0 <= 0.0:
            raise ValueError("alpha0 and beta0 must be positive")
        if share_negative < 0.0:
            raise ValueError("share_negative must be non-negative")
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.share_negative = float(share_negative)
        self.respect_freeze = bool(respect_freeze)

    def update_from_evidence(
        self,
        hypergraph: HypergraphOps,
        hyperpaths: Sequence[Hyperpath],
        sign: int,  # +1 for reward/confirm, -1 for error/depress
        now_t: Timestamp,
        state: PlasticityState,
    ) -> Mapping[EdgeId, ReliabilityScore]:
        if self.respect_freeze and getattr(state, "freeze", False):
            return {}

        # 1) Aggregate edge contributions from hyperpaths (EMA-consistent)
        contrib: Dict[EdgeId, float] = {}
        total = 0.0
        for hp in hyperpaths:
            w_p = float(max(0.0, hp.score))
            if w_p == 0.0:
                continue
            total += w_p
            for eid in hp.edges:
                contrib[eid] = contrib.get(eid, 0.0) + w_p

        if total <= 0.0:
            return {}

        updates: Dict[EdgeId, ReliabilityScore] = {}

        # 2) For each edge, compute normalized contribution and posterior mean
        for eid, s in contrib.items():
            e = hypergraph.get_edge(eid)
            if e is None:
                continue

            s_norm = float(s / total)

            # Existing integer counters
            s_base = float(e.counts_success)
            f_base = float(e.counts_miss)

            if sign > 0:
                s_add_f = s_norm
                f_add_f = 0.0
            elif sign < 0:
                s_add_f = 0.0
                f_add_f = self.share_negative * s_norm
            else:
                # sign == 0: no directional update
                continue

            s_post = s_base + s_add_f
            f_post = f_base + f_add_f

            # Posterior mean with priors
            num = self.alpha0 + s_post
            den = self.alpha0 + self.beta0 + s_post + f_post
            r_new = float(num / den) if den > 0.0 else float(e.reliability)
            r_new = float(state.clamp(r_new))

            # Commit reliability and timestamp
            e.reliability = r_new
            e.last_update_t = now_t

            # Integer counter updates (rounded contributions)
            s_inc = int(round(s_add_f))
            f_inc = int(round(f_add_f))
            if s_inc:
                e.counts_success += s_inc
            if f_inc:
                e.counts_miss += f_inc

            updates[eid] = r_new

        return updates

    def prune(
        self,
        hypergraph: HypergraphOps,
        now_t: Timestamp,
        state: PlasticityState,
    ) -> int:
        # Delegate pruning with current threshold
        return int(hypergraph.prune(now_t, float(state.prune_threshold)))


__all__ = ["BetaPlasticityEngine", "edge_posterior_params"]