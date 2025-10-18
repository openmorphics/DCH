# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Causal Rule Cards (CRC) extraction with Beta Monte Carlo reliability composition.

This module provides:
- CausalRuleCard: a dataclass summarizing a promoted causal rule
- CRCExtractor: constructs CausalRuleCard from a Hyperpath using Beta-Bernoulli
  posteriors for each constituent edge and Monte Carlo composition for the path.

Reuse:
- Interfaces: HypergraphOps, Hyperpath (see dch_core.interfaces)
- Beta utilities: posterior_params, sample_beta (see dch_core.beta_utils)
- Stdlib only: hashlib/json/datetime/os/typing

Notes:
- We treat the provided Hyperpath label as a human-readable string and avoid
  heavy parsing. NLG text embeds the label directly.
- Path reliability samples are computed as the product of per-edge Beta samples.
- The credible interval is computed directly from the path-level samples list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import hashlib
import math

from dch_core.interfaces import HypergraphOps, Hyperpath
from dch_core.beta_utils import posterior_params, sample_beta


@dataclass(frozen=True)
class CausalRuleCard:
    rule_id: str
    label: str
    type: str  # "Developing" | "Frozen"
    support: int | float
    reliability_mean: float
    reliability_ci: Tuple[float, float]
    text: str
    provenance_edges: List[str]


class CRCExtractor:
    def __init__(
        self,
        hypergraph: HypergraphOps,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        ci_level: float = 0.95,
        samples: int = 5000,
    ) -> None:
        if alpha0 <= 0.0 or beta0 <= 0.0:
            raise ValueError("alpha0 and beta0 must be positive")
        if not (0.0 < ci_level < 1.0):
            raise ValueError("ci_level must be in (0,1)")
        if int(samples) <= 0:
            raise ValueError("samples must be positive")
        self.graph = hypergraph
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.ci_level = float(ci_level)
        self.samples = int(samples)

    def make_card(
        self,
        label: str,
        hyperpath: Hyperpath,
        support: float | int,
        frozen_threshold: float = 0.9,
    ) -> CausalRuleCard:
        """
        Build a CausalRuleCard:
        - For each edge in hyperpath:
            * compute Beta posterior params via posterior_params(edge, alpha0, beta0)
            * sample reliability via sample_beta(...)
        - Path reliability samples := product over edges of per-edge samples (index-wise)
        - Compute mean and credible interval from path samples
        - Rule 'type' = "Frozen" if mean >= frozen_threshold else "Developing"
        - rule_id is a deterministic SHA-256 hexdigest (truncated) of label + edge ids
        """
        # Collect per-edge MC samples
        per_edge_samples: List[List[float]] = []
        present_edge_ids: List[str] = []

        for eid in hyperpath.edges:
            e = self.graph.get_edge(eid)  # may be None if missing
            if e is None:
                continue
            a_post, b_post = posterior_params(e, self.alpha0, self.beta0)
            # Draw MC samples for this edge
            s = sample_beta(a_post, b_post, n=self.samples, seed=None)
            # Clamp samples into [0,1] defensively
            s = [max(0.0, min(1.0, float(x))) for x in s]
            per_edge_samples.append(s)
            present_edge_ids.append(str(eid))

        # If no edges found (degenerate), fallback to neutral reliability of 1.0
        if not per_edge_samples:
            path_samples = [1.0] * self.samples
        else:
            # Compose index-wise product across edges
            # All lists have length self.samples
            path_samples = [1.0] * self.samples
            for s in per_edge_samples:
                for i in range(self.samples):
                    path_samples[i] *= s[i]
            # Clamp numerically
            path_samples = [max(0.0, min(1.0, float(x))) for x in path_samples]

        # Summary statistics from path-level MC
        mean_val = float(sum(path_samples) / float(len(path_samples)))
        lo, hi = self._credible_interval_from_samples(path_samples, self.ci_level)

        card_type = "Frozen" if mean_val >= float(frozen_threshold) else "Developing"

        # Deterministic rule id from label + edge ids
        h = hashlib.sha256()
        h.update(str(label).encode("utf-8"))
        h.update(b"|")
        for eid_str in present_edge_ids:
            h.update(str(eid_str).encode("utf-8"))
            h.update(b"|")
        rule_id = h.hexdigest()[:16]

        # NLG text (lightweight; embed label and formatted stats)
        ci_pct = int(round(self.ci_level * 100))
        text = (
            f"IF {label} THEN rule with mean={mean_val:.2%}, "
            f"CI_{ci_pct}=[{lo:.2%},{hi:.2%}], support={support}"
        )

        return CausalRuleCard(
            rule_id=rule_id,
            label=str(label),
            type=card_type,
            support=support,
            reliability_mean=mean_val,
            reliability_ci=(lo, hi),
            text=text,
            provenance_edges=list(present_edge_ids),
        )

    # ---------- helpers ----------

    @staticmethod
    def _quantile(sorted_x: Sequence[float], p: float) -> float:
        """
        Linear-interpolated quantile for 0 <= p <= 1 on a pre-sorted non-empty sequence.
        Matches the 'linear' method used in beta_utils for consistency.
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

    def _credible_interval_from_samples(self, samples: Sequence[float], level: float) -> Tuple[float, float]:
        xs = sorted(float(x) for x in samples)
        tail = (1.0 - float(level)) / 2.0
        p_lo = max(0.0, min(1.0, tail))
        p_hi = max(0.0, min(1.0, 1.0 - tail))
        lo = self._quantile(xs, p_lo)
        hi = self._quantile(xs, p_hi)
        # Clamp
        lo = max(0.0, min(1.0, float(lo)))
        hi = max(0.0, min(1.0, float(hi)))
        if lo > hi:
            lo, hi = float(xs[0]), float(xs[-1])
        return float(lo), float(hi)


__all__ = ["CausalRuleCard", "CRCExtractor"]