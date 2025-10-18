# -*- coding: utf-8 -*-
"""Task-Aware Scaffolding policy engine (standalone, torch-free).

This module implements a simple, deterministic policy engine for deciding whether to
reuse or isolate structure for a new task, plan freezing of critical edges, and emit
region tags for biasing structural growth. It does not depend on any graph backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple, Optional, List, Literal
import math

# Public actions (enum-like)
ISOLATE: Literal["ISOLATE"] = "ISOLATE"
REUSE: Literal["REUSE"] = "REUSE"
Action = Literal["ISOLATE", "REUSE"]


@dataclass(frozen=True)
class ScaffoldingParams:
    """Configuration parameters for the scaffolding policy engine.

    Fields:
        similarity_metric: Only "jaccard" is supported.
        similarity_threshold: Similarity threshold; >= threshold => REUSE else ISOLATE.
        freeze_strategy: Either "resistance" or "immutable".
        freeze_strength: Resistance value in [0,1] used when freeze_strategy=="resistance".
        freeze_top_k: Top fraction (0,1] of eligible edges to freeze.
        reliability_cutoff: Only edges with reliability >= cutoff are considered eligible.
        region_prefix: Region tag prefix, e.g., "task".
        epsilon: Numerical tolerance to stabilize comparisons.
    """

    similarity_metric: str = "jaccard"
    similarity_threshold: float = 0.5
    freeze_strategy: str = "resistance"  # {"resistance","immutable"}
    freeze_strength: float = 1.0         # used when freeze_strategy=="resistance"
    freeze_top_k: float = 0.2            # (0,1]
    reliability_cutoff: float = 0.6      # [0,1]
    region_prefix: str = "task"
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.similarity_metric != "jaccard":
            raise ValueError("Only 'jaccard' similarity is supported")
        if self.freeze_strategy not in ("resistance", "immutable"):
            raise ValueError("freeze_strategy must be one of {'resistance','immutable'}")
        if not (0.0 <= self.freeze_strength <= 1.0):
            raise ValueError("freeze_strength must be in [0,1]")
        if not (0.0 < self.freeze_top_k <= 1.0):
            raise ValueError("freeze_top_k must be in (0,1]")
        if not (0.0 <= self.reliability_cutoff <= 1.0):
            raise ValueError("reliability_cutoff must be in [0,1]")
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in [0,1]")
        if not (self.epsilon > 0.0):
            raise ValueError("epsilon must be > 0")


class DefaultScaffoldingPolicy:
    """Default task-aware scaffolding policy.

    The policy:
      - Computes task similarity (Jaccard) from activation signatures.
      - Decides REUSE vs ISOLATE using a threshold on similarity.
      - Plans freezing by selecting the top-k fraction of reliable edges.
      - Applies update resistance (immutable or scaled).
      - Emits deterministic region tags.

    Determinism:
      - Sorting is stable and ties are broken deterministically by (-reliability, edge_id).
      - For decision ties on similarity, lexicographically smallest task_id is chosen.
    """

    def __init__(self, params: ScaffoldingParams):
        """Initialize the policy with the given parameters."""
        self.params = params
        self._tasks_signatures: Dict[str, Set[str]] = {}
        self._tasks_reliability: Dict[str, Dict[str, float]] = {}

    def register_task(
        self,
        task_id: str,
        activation_signature: Set[str],
        edge_reliability: Dict[str, float],
    ) -> None:
        """Register a task's activation signature and reliability snapshot.

        Args:
            task_id: Unique task identifier.
            activation_signature: Set of activated edge IDs for this task.
            edge_reliability: Mapping edge_id -> reliability score in [0,1].

        Notes:
            - Reliability snapshot is normalized and clamped to [0,1].
            - Only edges present in the signature are stored in the snapshot.
            - Re-registering a task replaces previous entries.
        """
        # Keep a copy to avoid external mutation
        sig = set(activation_signature or set())
        # Normalize reliability constrained to edges in signature; missing -> 0.0
        rel_snapshot: Dict[str, float] = {}
        for eid in sig:
            r = float(edge_reliability.get(eid, 0.0))
            if r < 0.0:
                r = 0.0
            elif r > 1.0:
                r = 1.0
            rel_snapshot[eid] = r
        self._tasks_signatures[task_id] = sig
        self._tasks_reliability[task_id] = rel_snapshot

    def similarity(self, signature_a: Set[str], signature_b: Set[str]) -> float:
        """Compute Jaccard similarity between two activation signatures.

        Returns:
            |A ∩ B| / |A ∪ B|. If both are empty, returns 0.0.
        """
        if not signature_a and not signature_b:
            return 0.0
        inter = len(signature_a & signature_b)
        union = len(signature_a | signature_b)
        if union == 0:
            return 0.0
        return inter / float(union)

    def decide(self, new_signature: Set[str]) -> Tuple[Action, Optional[str]]:
        """Decide whether to REUSE an existing task's structure or ISOLATE for the new task.

        Args:
            new_signature: Activation signature (set of edge IDs) for the new task.

        Returns:
            (action, closest_task_id_or_None)
        """
        cutoff = self.params.reliability_cutoff
        # Pre-filter stored signatures by reliability cutoff
        candidates: List[Tuple[str, Set[str]]] = []
        for tid, sig in self._tasks_signatures.items():
            rel_map = self._tasks_reliability.get(tid, {})
            filtered = {e for e in sig if rel_map.get(e, 0.0) + self.params.epsilon >= cutoff}
            candidates.append((tid, filtered))

        best_tid: Optional[str] = None
        best_sim: float = -1.0
        # Sort by task_id for deterministic tie-breaking
        for tid, sig in sorted(candidates, key=lambda kv: kv[0]):
            sim = self.similarity(sig, new_signature)
            if sim > best_sim + self.params.epsilon:
                best_sim = sim
                best_tid = tid
            # else: tie retains earlier (lexicographically smaller) tid

        if best_sim + self.params.epsilon >= self.params.similarity_threshold and best_tid is not None:
            return REUSE, best_tid
        return ISOLATE, None

    def plan_freeze(self, base_task_id: str) -> Dict[str, float]:
        """Plan a freeze mask for a base task.

        Selects the top_k fraction (ceil) of edges among those that:
          - belong to the base task's signature; and
          - have reliability >= reliability_cutoff.

        Returns:
            Mapping edge_id -> resistance (0..1). For "immutable" strategy,
            resistance is 1.0 for selected edges. For "resistance" strategy,
            resistance is params.freeze_strength.
        """
        if base_task_id not in self._tasks_signatures:
            raise KeyError(f"Unknown task_id: {base_task_id}")

        cutoff = self.params.reliability_cutoff
        sig = self._tasks_signatures[base_task_id]
        rel_map = self._tasks_reliability.get(base_task_id, {})

        eligible: List[Tuple[str, float]] = []
        for e in sig:
            r = float(rel_map.get(e, 0.0))
            if r + self.params.epsilon >= cutoff:
                eligible.append((e, r))

        if not eligible:
            return {}

        # Deterministic sort: by (-reliability, edge_id)
        eligible.sort(key=lambda pair: (-pair[1], pair[0]))

        n = len(eligible)
        k = int(math.ceil(n * self.params.freeze_top_k))
        k = max(0, min(n, k))
        selected = eligible[:k]

        resistance_value = 1.0 if self.params.freeze_strategy == "immutable" else float(self.params.freeze_strength)
        if resistance_value < 0.0:
            resistance_value = 0.0
        elif resistance_value > 1.0:
            resistance_value = 1.0

        return {eid: resistance_value for eid, _ in selected}

    def apply_update_resistance(self, edge_id: str, delta: float, freeze_mask: Dict[str, float]) -> float:
        """Apply update resistance to a delta for an edge.

        For "immutable" strategy, returns 0.0 if edge_id is frozen.
        Otherwise, returns delta * (1 - resistance), where resistance is
        taken from the freeze_mask (defaults to 0.0 if missing).
        """
        if self.params.freeze_strategy == "immutable" and edge_id in freeze_mask:
            return 0.0
        r = float(freeze_mask.get(edge_id, 0.0))
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        return delta * (1.0 - r)

    def region_tag(self, new_task_id: str) -> str:
        """Return deterministic region tag '<prefix>:<new_task_id>'."""
        return f"{self.params.region_prefix}:{new_task_id}"


__all__ = ["ScaffoldingParams", "DefaultScaffoldingPolicy", "ISOLATE", "REUSE"]