# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Hierarchical abstraction engine (torch-free) for promoting frequent hyperpaths
to higher-order hyperedges (HOEs) within an in-memory hypergraph backend.

This module provides:
- AbstractionParams: small config dataclass
- DefaultAbstractionEngine: concrete engine with a direct API promote(hyperpath)

Key behaviors
- Head of the new HOE is the sink vertex of the hyperpath
- Tail is the set of source vertices (those that never appear as a head within the hyperpath)
- Temporal params (delta_min/delta_max) derived from observed delays Δt = head_t - tail_t across tails
- Reliability initialized from constituent hyperedges via an aggregation mode
- Provenance optionally stored in the new edge attributes
- Deduplication on (head, sorted(tail)) with idempotence and optional reliability bump

Notes
- Compatible with in-memory HypergraphOps backend (dch_core.hypergraph_mem.InMemoryHypergraph)
- No torch dependency; only Python stdlib
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Mapping

import math

from dch_core.interfaces import (
    EdgeId,
    Hyperedge,
    HypergraphOps,
    Hyperpath,
    ReliabilityScore,
    Timestamp,
    Vertex,
    VertexId,
    make_edge_id,
)


@dataclass
class AbstractionParams:
    """
    Configuration for higher-order hyperedge promotion.

    Fields
    - reliability_agg: aggregation mode for initializing reliability of the new HOE
        * "min": min of constituent reliabilities
        * "mean": arithmetic mean
        * "geo": geometric mean
    - reliability_floor: lower bound to avoid zero initialization
    - provenance: if True, store provenance payload in edge.attributes["provenance"]
    - dedup: if True, avoid duplicates on (head, sorted(tail)) and be idempotent
    """
    reliability_agg: str = "min"  # {"min","mean","geo"}
    reliability_floor: float = 0.10
    provenance: bool = True
    dedup: bool = True


class DefaultAbstractionEngine:
    """
    Concrete abstraction engine with a direct API for promoting a provided Hyperpath.

    Usage:
        engine = DefaultAbstractionEngine(graph_ops, AbstractionParams())
        new_eid = engine.promote(hyperpath)

    Methods:
    - promote(hyperpath) -> str: create (or reuse) a HOE and return its id
    - exists(head_vertex_id, tail_vertex_ids) -> Optional[str]: check for an existing HOE
    - make_label(hyperpath) -> str: canonical label stable to tail permutations and time shifts
    """

    def __init__(self, graph_ops: HypergraphOps, params: AbstractionParams) -> None:
        self.graph = graph_ops
        self.params = params

    # --------------- Public API ---------------

    def promote(self, hyperpath: Hyperpath) -> str:
        """
        Promote the given hyperpath into a higher-order hyperedge (HOE).

        Returns:
            EdgeId as a string (existing or newly created).
        """
        head_vid: VertexId = hyperpath.head
        head_v: Optional[Vertex] = self.graph.get_vertex(head_vid)
        if head_v is None:
            raise ValueError(f"Hyperpath head vertex not found: {head_vid}")

        # Compute sources (tail set) from the hyperpath structure
        sources: Set[VertexId] = self._compute_sources(hyperpath)
        if not sources:
            # Degenerate path (no constituent edges or sources) -> nothing to promote
            raise ValueError("Cannot promote hyperpath with empty source set")

        # Dedup check (idempotent)
        if self.params.dedup:
            existing = self.exists(str(head_vid), tuple(sorted(str(v) for v in sources)))
            if existing is not None:
                # Optional reliability bump if aggregated is higher
                self._maybe_bump_reliability(existing, hyperpath)
                return existing

        # Temporal parameters from observed delays Δt = head_t - tail_t
        deltas = self._observed_deltas(head_v.t, sources)
        delta_min = min(deltas)
        delta_max = max(deltas)
        refractory_rho = self._infer_refractory_rho(head_vid)

        # Reliability initialization
        init_rel = self._aggregate_reliability(hyperpath)
        init_rel = self._clamp_reliability(init_rel, self.params.reliability_floor, 1.0)

        # Build provenance attributes if enabled
        attributes: Dict[str, object] = {}
        if self.params.provenance:
            attributes["provenance"] = {
                "type": "abstraction",
                "from_edges": [str(eid) for eid in hyperpath.edges],
                "hyperpath_label": self.make_label(hyperpath),
            }

        # Compose deterministic id using canonical tail sort and head time as nonce
        eid: EdgeId = make_edge_id(head=head_vid, tail=sources, t=head_v.t)
        hoe = Hyperedge(
            id=eid,
            tail=set(sources),
            head=head_vid,
            delta_min=int(delta_min),
            delta_max=int(delta_max),
            refractory_rho=int(refractory_rho),
            reliability=float(init_rel),
            provenance="abstraction",
            attributes=attributes,
        )

        admitted = list(self.graph.insert_hyperedges([hoe]))
        if not admitted:
            # Another concurrent insert could have deduped it; resolve id
            existing = self.exists(str(head_vid), tuple(sorted(str(v) for v in sources)))
            if existing is None:
                # Backend must have rejected for another reason
                raise RuntimeError("Failed to insert higher-order hyperedge and no duplicate found")
            # Optional reliability bump to reflect current aggregate evidence
            self._maybe_bump_reliability(existing, hyperpath)
            return existing

        return str(admitted[0])

    def exists(self, head_vertex_id: str, tail_vertex_ids: Tuple[str, ...]) -> Optional[str]:
        """
        Check whether a hyperedge with identical (head, sorted tail) already exists.

        Args:
            head_vertex_id: head vertex id string
            tail_vertex_ids: tail vertex id strings (order will be ignored)

        Returns:
            Existing EdgeId as string if found, else None.
        """
        head_v = VertexId(head_vertex_id)
        tail_set: Set[VertexId] = {VertexId(t) for t in tail_vertex_ids}
        for eid in self.graph.get_incoming_edges(head_v):
            e = self.graph.get_edge(eid)
            if e is None:
                continue
            if e.head == head_v and set(e.tail) == tail_set:
                return str(e.id)
        return None

    def make_label(self, hyperpath: Hyperpath) -> str:
        """
        Construct a canonical label string for the given hyperpath that is:
        - invariant to permutation of tails within each constituent edge
        - invariant to absolute time shifts (uses Δt = head_time - tail_time)

        Format (informal):
            "HEAD:{sink_neuron}|EDGES:[(h, ((n,dt),...)), ...]"
        where dt are integer deltas derived from vertex ids if canonical, else 0.
        """
        sink_neu, _ = self._parse_vertex_id(hyperpath.head)
        edge_tokens: List[str] = []

        for eid in hyperpath.edges:
            parsed = self._parse_edge_id(eid)
            if parsed is None:
                edge_tokens.append(f"RAW:{str(eid)}")
                continue
            h_vid, tails = parsed
            h_neu, h_t = self._parse_vertex_id(h_vid)
            pairs: List[Tuple[str, int]] = []
            for tv in tails:
                t_neu, t_t = self._parse_vertex_id(tv)
                dt = self._delta_t(h_t, t_t)
                pairs.append((str(t_neu), int(dt)))
            pairs.sort()
            edge_tokens.append(repr((str(h_neu), tuple(pairs))))

        edge_tokens.sort()
        return f"HEAD:{sink_neu}|EDGES:[{','.join(edge_tokens)}]"

    # --------------- Internals ---------------

    def _compute_sources(self, hyperpath: Hyperpath) -> Set[VertexId]:
        """Sources are vertices that appear in some tail but never as a head within the hyperpath."""
        heads: Set[VertexId] = set()
        tails: Set[VertexId] = set()
        for eid in hyperpath.edges:
            e = self.graph.get_edge(eid)
            if e is None:
                # If edges are missing, skip them; sources are derived from present edges
                continue
            heads.add(e.head)
            tails.update(set(e.tail))
        return {tv for tv in tails if tv not in heads}

    def _observed_deltas(self, head_time: Timestamp, tails: Set[VertexId]) -> List[int]:
        """Compute observed Δt = head_time - tail_time for each tail vertex."""
        deltas: List[int] = []
        for tv in tails:
            v = self.graph.get_vertex(tv)
            if v is None:
                continue
            deltas.append(int(head_time) - int(v.t))
        if not deltas:
            # Fallback to [0] to avoid invalid Hyperedge (will be clamped by __post_init__)
            deltas = [0]
        return deltas

    def _infer_refractory_rho(self, head: VertexId) -> int:
        """
        Re-use a refractory value if available from any existing incoming edge for the same head.
        Otherwise default to 0 (no refractory).
        """
        for eid in self.graph.get_incoming_edges(head):
            e = self.graph.get_edge(eid)
            if e is not None:
                try:
                    return int(e.refractory_rho)
                except Exception:
                    continue
        return 0

    def _aggregate_reliability(self, hyperpath: Hyperpath) -> ReliabilityScore:
        """Aggregate reliabilities across constituent hyperedges per configured mode."""
        rs: List[float] = []
        for eid in hyperpath.edges:
            e = self.graph.get_edge(eid)
            if e is None:
                continue
            try:
                r = float(e.reliability)
            except Exception:
                r = 0.0
            # Clamp into [0,1] just in case
            r = min(1.0, max(0.0, r))
            rs.append(r)

        if not rs:
            return float(self.params.reliability_floor)

        mode = (self.params.reliability_agg or "min").lower()
        if mode == "min":
            agg = min(rs)
        elif mode == "mean":
            agg = sum(rs) / float(len(rs))
        elif mode == "geo":
            # Robust geometric mean; zero if any r == 0
            if any(r <= 0.0 for r in rs):
                agg = 0.0
            else:
                prod = 1.0
                for r in rs:
                    prod *= float(r)
                agg = prod ** (1.0 / float(len(rs)))
        else:
            raise ValueError(f"Unsupported reliability_agg mode: {self.params.reliability_agg}")
        return float(agg)

    def _clamp_reliability(self, r: float, lo: float, hi: float) -> float:
        """Clamp reliability into [lo, hi]."""
        return max(float(lo), min(float(hi), float(r)))

    def _maybe_bump_reliability(self, eid_str: str, hyperpath: Hyperpath, eps: float = 1e-6) -> None:
        """If aggregated reliability > current + eps, bump the stored reliability to the new value."""
        e = self.graph.get_edge(EdgeId(eid_str))
        if e is None:
            return
        new_rel = self._clamp_reliability(self._aggregate_reliability(hyperpath), self.params.reliability_floor, 1.0)
        if float(new_rel) > float(e.reliability) + float(eps):
            e.reliability = float(new_rel)

    # --- canonical label helpers (WL-style invariance) ---

    def _delta_t(self, head_time: Optional[int], tail_time: Optional[int]) -> int:
        if head_time is None or tail_time is None:
            return 0
        return int(head_time) - int(tail_time)

    def _parse_vertex_id(self, vid: VertexId) -> Tuple[str, Optional[int]]:
        """Parse 'neuron@time' -> (neuron_str, time_int or None)."""
        s = str(vid)
        if "@" not in s:
            return s, None
        neu_s, t_s = s.split("@", 1)
        try:
            t = int(t_s)
        except Exception:
            t = None
        return neu_s, t

    def _parse_edge_id(self, eid: EdgeId) -> Optional[Tuple[VertexId, List[VertexId]]]:
        """
        Parse canonical-like edge id of form:
            'HEAD&tail1,tail2,...#nonce'
        Returns (head_vid, [tail_vids]) or None if parsing fails.
        """
        s = str(eid)
        left, _sep_hash, _nonce = s.partition("#")
        head_s, sep, tails_s = left.partition("&")
        if sep == "":
            return None
        tails = [VertexId(x) for x in tails_s.split(",") if x]
        return VertexId(head_s), tails


__all__ = ["AbstractionParams", "DefaultAbstractionEngine"]