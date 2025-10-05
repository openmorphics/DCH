# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Constrained backward hyperpath traversal (beam search with AND-frontier).

Implements a default TraversalEngine that:
- Starts from a target vertex (head event)
- Expands backwards along admissible incoming hyperedges
- Enforces B-connectivity by replacing a frontier vertex with the full tail of a traversed hyperedge
- Respects temporal admissibility windows and a global backward horizon
- Scores paths by composing edge reliabilities with an optional length penalty
- Maintains a beam of top-K partial paths to bound branching

References:
- Contracts: dch_core.interfaces (TraversalEngine, HypergraphOps, Hyperpath, etc.)
- Algorithm specs: docs/AlgorithmSpecs.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from dch_core.interfaces import (
    EdgeId,
    Hyperedge,
    HypergraphOps,
    Hyperpath,
    Timestamp,
    Vertex,
    VertexId,
    is_temporally_admissible,
)


# -------------------------
# Utilities
# -------------------------


def _length_penalty(length: int, base: float = 0.98) -> float:
    """Monotone penalty to discourage overly long paths."""
    return base**length


def _canonical_label(edges: Sequence[EdgeId]) -> str:
    """
    Stable canonical label for a hyperpath.
    Uses lexical sort over edge ids; sufficient for deduplication at this stage.
    """
    return "|".join(sorted(edges))


# -------------------------
# Beam search state
# -------------------------


@dataclass(frozen=True)
class _PathState:
    """Partial path with an AND-frontier of unresolved vertices."""
    head: VertexId
    head_time: Timestamp
    edges: Tuple[EdgeId, ...]  # traversed edges so far (backward)
    score: float               # composed reliability with penalties
    frontier: Tuple[VertexId, ...]  # unresolved vertices to expand (AND frontier)

    def extend(self, add_edge: EdgeId, add_score: float, replace_vid: VertexId, new_tail: Sequence[VertexId]) -> "_PathState":
        """Return a new state by replacing one frontier vertex with the edge tail."""
        # Build new frontier: remove replace_vid, add new_tail (dedup while preserving order)
        new_frontier_list: List[VertexId] = [vid for vid in self.frontier if vid != replace_vid]
        # Append tail (can include duplicates across expansions; remove duplicates)
        for vid in new_tail:
            if vid not in new_frontier_list:
                new_frontier_list.append(vid)
        return _PathState(
            head=self.head,
            head_time=self.head_time,
            edges=self.edges + (add_edge,),
            score=self.score * add_score,
            frontier=tuple(new_frontier_list),
        )


# -------------------------
# Default traversal engine
# -------------------------


class DefaultTraversalEngine:
    """
    Default implementation of constrained backward traversal with an AND-frontier.

    Notes:
    - Beam selection is deterministic by default (top-K by score).
    - rng parameter is reserved for future randomized branching policies.
    """

    def __init__(self, length_penalty_base: float = 0.98) -> None:
        self.length_penalty_base = length_penalty_base

    # API matches TraversalEngine Protocol
    def backward_traverse(
        self,
        hypergraph: HypergraphOps,
        target: Vertex,
        horizon: int,
        beam_size: int,
        rng: Optional[Callable[[int], float]],
        refractory_enforce: bool = True,  # kept for signature parity; refractory is handled by edge temporal constraints here
    ) -> Sequence[Hyperpath]:
        if beam_size <= 0:
            return []

        # Initialize beam with target vertex as the only unresolved frontier
        beam: List[_PathState] = [
            _PathState(
                head=target.id,
                head_time=target.t,
                edges=tuple(),
                score=1.0,
                frontier=(target.id,),
            )
        ]
        results: List[Hyperpath] = []

        # To prevent pathological looping, keep a conservative step bound
        max_steps = max(1, beam_size * 8)

        steps = 0
        while beam and steps < max_steps:
            steps += 1
            next_beam: List[_PathState] = []

            for state in beam:
                if not state.frontier:
                    # Fully resolved path (no unresolved vertices)
                    results.append(
                        Hyperpath(
                            head=state.head,
                            edges=state.edges,
                            score=state.score * _length_penalty(len(state.edges), self.length_penalty_base),
                            length=len(state.edges),
                            label=_canonical_label(state.edges),
                        )
                    )
                    continue

                # Choose one frontier vertex to expand (greedy: latest in time)
                expand_vid = state.frontier[-1]
                expand_vertex = hypergraph.get_vertex(expand_vid)
                if expand_vertex is None:
                    # Missing vertex; treat as source and finalize this branch
                    results.append(
                        Hyperpath(
                            head=state.head,
                            edges=state.edges,
                            score=state.score * _length_penalty(len(state.edges), self.length_penalty_base),
                            length=len(state.edges),
                            label=_canonical_label(state.edges),
                        )
                    )
                    continue

                # Fetch admissible incoming edges for expand_vid under temporal logic + horizon
                incoming_eids = list(hypergraph.get_incoming_edges(expand_vid))
                if not incoming_eids:
                    # No edges to expand; keep as a leaf (source)
                    results.append(
                        Hyperpath(
                            head=state.head,
                            edges=state.edges,
                            score=state.score * _length_penalty(len(state.edges), self.length_penalty_base),
                            length=len(state.edges),
                            label=_canonical_label(state.edges),
                        )
                    )
                    continue

                expanded = False
                for eid in incoming_eids:
                    e = hypergraph.get_edge(eid)
                    if e is None:
                        continue
                    # Gather tail vertex times; reject if any tail vertex missing
                    tail_vertices: List[Vertex] = []
                    missing_tail = False
                    for tvid in e.tail:
                        tv = hypergraph.get_vertex(tvid)
                        if tv is None:
                            missing_tail = True
                            break
                        tail_vertices.append(tv)
                    if missing_tail:
                        continue

                    # Temporal admissibility relative to the head of this subproblem (expand_vertex time)
                    tail_times = [tv.t for tv in tail_vertices]
                    if not is_temporally_admissible(tail_times, expand_vertex.t, e.delta_min, e.delta_max):
                        continue

                    # Global horizon: ensure the earliest tail is not older than target.t - horizon
                    if horizon > 0:
                        earliest_tail = min(tail_times)
                        if (target.t - earliest_tail) > horizon:
                            continue

                    # Accept this expansion; compute score increment with reliability
                    edge_contrib = max(1e-6, float(e.reliability))
                    new_state = state.extend(
                        add_edge=e.id,
                        add_score=edge_contrib,
                        replace_vid=expand_vid,
                        new_tail=[tv.id for tv in tail_vertices],
                    )
                    next_beam.append(new_state)
                    expanded = True

                if not expanded:
                    # If nothing was admissible, finalize as a leaf path
                    results.append(
                        Hyperpath(
                            head=state.head,
                            edges=state.edges,
                            score=state.score * _length_penalty(len(state.edges), self.length_penalty_base),
                            length=len(state.edges),
                            label=_canonical_label(state.edges),
                        )
                    )

            # Beam select for next iteration
            if not next_beam:
                break
            next_beam.sort(key=lambda st: st.score * _length_penalty(len(st.edges), self.length_penalty_base), reverse=True)
            beam = next_beam[:beam_size]

        # Deduplicate by canonical label and keep best-scoring instance
        dedup: Dict[str, Hyperpath] = {}
        for hp in results:
            key = hp.label or _canonical_label(hp.edges)
            prev = dedup.get(key)
            if prev is None or hp.score > prev.score:
                dedup[key] = hp

        return list(dedup.values())


__all__ = ["DefaultTraversalEngine"]