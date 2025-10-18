# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Streaming Frequent Hyperpath Miner (FSM) for DCH.

Implements:
- StreamingFSMEngine: a CPU-friendly online miner that:
  * accepts batches of Hyperpath objects
  * canonicalizes labels (uses provided Hyperpath.label or a deterministic fallback)
  * maintains exponentially decayed counts over event time
  * applies a frequency threshold with simple hysteresis before promotion

Design notes
- Decay: count[label] := count[label] * exp(-lambda * dt) + weight(p)
  where dt = max(0, now_t - last_update_t[label]) in the same time units as timestamps.
- Weight: defaults to hyperpath score (>= 0). If absent, weight = 1.0.
- Hysteresis: require at least 'hold_k' total observations for the label; promotion occurs
  when the current (decayed) count >= theta and the observation count >= hold_k.

References
- Protocol: dch_core.interfaces.FSMEngine
- Canonicalization: docs/AlgorithmSpecs.md (labels)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple, List

from dch_core.interfaces import FSMEngine, Hyperpath, Timestamp


def _fallback_label(hp: Hyperpath) -> str:
    """Deterministic fallback canonical label from edge ids."""
    return "|".join(sorted(hp.edges))


@dataclass
class _State:
    count: float = 0.0
    last_t: Optional[Timestamp] = None
    hold: int = 0  # hysteresis counter


@dataclass
class StreamingFSMEngine(FSMEngine):
    """
    Streaming hyperpath miner with exponential decay and hysteresis.

    Parameters
    - theta: promotion threshold on (decayed) count
    - lambda_decay: exponential decay rate per unit time (0 => no decay)
    - hold_k: require K consecutive observes above threshold before promotion
    - min_weight: lower bound for per-hyperpath weight to avoid underflow
    """

    theta: float = 5.0
    lambda_decay: float = 0.0
    hold_k: int = 2
    min_weight: float = 1e-6

    _table: Dict[str, _State] = field(default_factory=dict)
    _promotions_queue: List[str] = field(default_factory=list)

    def _decay_then_add(self, label: str, weight: float, now_t: Timestamp) -> float:
        st = self._table.get(label)
        if st is None:
            st = _State(count=0.0, last_t=None, hold=0)
            self._table[label] = st
        # Decay
        if self.lambda_decay > 0.0 and st.last_t is not None:
            dt = max(0, int(now_t) - int(st.last_t))
            if dt > 0:
                st.count *= math.exp(-self.lambda_decay * float(dt))
        # Add
        st.count += max(self.min_weight, float(weight))
        st.last_t = now_t
        return st.count

    def observe(
        self,
        hyperpaths: Sequence[Hyperpath],
        now_t: Timestamp,
    ) -> Sequence[str]:
        promotions: List[str] = []

        for hp in hyperpaths:
            label = hp.label or _fallback_label(hp)
            weight = max(self.min_weight, float(hp.score if hp.score is not None else 1.0))
            cnt = self._decay_then_add(label, weight, now_t)

            st = self._table[label]
            # Count total observations for hysteresis; promotion requires both:
            # (a) accumulated (decayed) count >= theta and (b) at least hold_k observations
            st.hold += 1
            if cnt >= self.theta and st.hold >= self.hold_k:
                promotions.append(label)
                # Optional: reset observation counter to require another warm-up before next promotion
                st.hold = 0

        # Cache promotions for retrieval via pop_promotions()
        if promotions:
            self._promotions_queue.extend(promotions)

        return promotions


    def pop_promotions(self) -> List[str]:
        """Return and clear labels promoted since the last call."""
        out = list(self._promotions_queue)
        self._promotions_queue.clear()
        return out


__all__ = ["StreamingFSMEngine"]