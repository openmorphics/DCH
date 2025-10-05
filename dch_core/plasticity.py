# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Evidence-based plasticity engine for Dynamic Causal Hypergraph (DCH).

Implements:
- DefaultPlasticityEngine.update_from_evidence: discrete, evidence-based reliability updates
  using hyperpath scores with EMA and clamping from PlasticityState
- DefaultPlasticityEngine.prune: delegates to HypergraphOps.prune with threshold from state

References:
- Protocols and entities: dch_core.interfaces
- Algorithm specs: docs/AlgorithmSpecs.md (Section 3: Temporal Credit Assignment and Evidence Aggregation)
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

from dch_core.interfaces import (
    EdgeId,
    HypergraphOps,
    Hyperpath,
    PlasticityEngine,
    PlasticityState,
    ReliabilityScore,
    Timestamp,
)


class DefaultPlasticityEngine(PlasticityEngine):
    """
    Default implementation of DCH plasticity:
    - Aggregates hyperpath evidence per edge
    - Applies normalized contributions with EMA and clamping
    - Updates success/miss counters and last_update_t
    - Pruning is delegated to the hypergraph backend (budget-aware)
    """

    def update_from_evidence(
        self,
        hypergraph: HypergraphOps,
        hyperpaths: Sequence[Hyperpath],
        sign: int,  # +1 for reward/confirm, -1 for error/depress
        now_t: Timestamp,
        state: PlasticityState,
    ) -> Mapping[EdgeId, ReliabilityScore]:
        # Aggregate edge contributions from hyperpaths
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

        alpha = float(state.ema_alpha)
        updates: Dict[EdgeId, ReliabilityScore] = {}

        for eid, s in contrib.items():
            e = hypergraph.get_edge(eid)
            if e is None:
                continue
            s_norm = float(s / total)
            r_old = float(e.reliability)
            # Target step in direction of sign, normalized by evidence share
            r_tgt = r_old + float(sign) * s_norm
            # EMA update with clamping
            r_new = (1.0 - alpha) * r_old + alpha * r_tgt
            r_new = float(state.clamp(r_new))
            e.reliability = r_new
            if sign > 0:
                e.counts_success += 1
            else:
                e.counts_miss += 1
            e.last_update_t = now_t
            updates[eid] = r_new

        # Note: Optional global decay over all edges requires hypergraph-wide iteration.
        # The HypergraphOps protocol does not expose edge iteration; decay can be
        # implemented by the backend or invoked via a periodic maintenance hook.

        return updates

    def prune(
        self,
        hypergraph: HypergraphOps,
        now_t: Timestamp,
        state: PlasticityState,
    ) -> int:
        # Delegate to backend implementation (budget-aware pruning if provided)
        return int(hypergraph.prune(now_t, float(state.prune_threshold)))


__all__ = ["DefaultPlasticityEngine"]
