# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
In-memory HypergraphOps backend and a simple GraphConnectivity implementation.

Provides:
- InMemoryHypergraph: a basic, single-process implementation of HypergraphOps
  suitable for CPU-based experimentation and tests.
- StaticGraphConnectivity: a minimal connectivity oracle backed by a user-supplied
  mapping from postsynaptic neuron id -> iterable of presynaptic neuron ids.

Notes
- This backend is not optimized for large-scale workloads; window queries are O(N).
- Deduplication is performed on (head, sorted(tail)) keys during insertion.
- Pruning iterates over all edges and updates adjacency indices.

References
- Protocols and entities: dch_core.interfaces
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from dch_core.interfaces import (
    EdgeId,
    GraphConnectivity,
    Hyperedge,
    HypergraphOps,
    HypergraphSnapshot,
    NeuronId,
    Timestamp,
    Vertex,
    VertexId,
    Window,
    make_vertex_id,
)


def _edge_key(head: VertexId, tail: Set[VertexId]) -> Tuple[VertexId, Tuple[VertexId, ...]]:
    """Canonical dedup key for an edge: (head, sorted(tail))."""
    return (head, tuple(sorted(tail)))


class InMemoryHypergraph(HypergraphOps):
    """
    In-memory implementation of HypergraphOps.

    Data structures
    - _vertices: VertexId -> Vertex
    - _edges: EdgeId -> Hyperedge
    - _incoming: VertexId -> Set[EdgeId]    (edges whose head == vid)
    - _outgoing: VertexId -> Set[EdgeId]    (edges where vid in tail)
    - _time_index: List[Vertex]             (append-only list of vertices; window_query is O(N) filter)
    - _edge_keys: Map[(head, sorted(tail)) -> EdgeId] to prevent duplicates
    """

    def __init__(self) -> None:
        self._vertices: Dict[VertexId, Vertex] = {}
        self._edges: Dict[EdgeId, Hyperedge] = {}
        self._incoming: Dict[VertexId, Set[EdgeId]] = defaultdict(set)
        self._outgoing: Dict[VertexId, Set[EdgeId]] = defaultdict(set)
        self._time_index: List[Vertex] = []
        self._edge_keys: Dict[Tuple[VertexId, Tuple[VertexId, ...]], EdgeId] = {}

    # ---- Vertex operations ----

    def ingest_event(self, event) -> Vertex:
        """Materialize an event as a Vertex and update indices."""
        vid = make_vertex_id(event.neuron_id, event.t)
        v = self._vertices.get(vid)
        if v is None:
            v = Vertex(id=vid, neuron_id=event.neuron_id, t=event.t)
            self._vertices[vid] = v
            self._time_index.append(v)
        return v

    def window_query(self, window: Window) -> Sequence[Vertex]:
        """Return vertices with t in [t0, t1]."""
        t0, t1 = window
        if t1 < t0:
            return []
        # Linear scan; acceptable for small/medium datasets in CPU experiments
        return [v for v in self._time_index if t0 <= v.t <= t1]

    def get_vertex(self, vid: VertexId) -> Optional[Vertex]:
        return self._vertices.get(vid)

    # ---- Edge operations ----

    def get_edge(self, eid: EdgeId) -> Optional[Hyperedge]:
        return self._edges.get(eid)

    def get_incoming_edges(self, vid: VertexId) -> Set[EdgeId]:
        return set(self._incoming.get(vid, set()))

    def get_outgoing_edges(self, vid: VertexId) -> Set[EdgeId]:
        return set(self._outgoing.get(vid, set()))

    def insert_hyperedges(self, candidates: Sequence[Hyperedge]) -> Sequence[EdgeId]:
        """Insert candidates with dedup on (head, sorted(tail)); update adjacencies."""
        admitted: List[EdgeId] = []
        for e in candidates:
            key = _edge_key(e.head, e.tail)
            if key in self._edge_keys:
                # Duplicate; skip
                continue
            # Insert
            self._edges[e.id] = e
            self._edge_keys[key] = e.id
            self._incoming[e.head].add(e.id)
            for tvid in e.tail:
                self._outgoing[tvid].add(e.id)
            admitted.append(e.id)
        return admitted

    def _remove_edge(self, eid: EdgeId) -> None:
        e = self._edges.pop(eid, None)
        if e is None:
            return
        key = _edge_key(e.head, e.tail)
        # delete edge key if it still points to eid
        if self._edge_keys.get(key) == eid:
            del self._edge_keys[key]
        # update adjacency
        if eid in self._incoming.get(e.head, set()):
            self._incoming[e.head].remove(eid)
            if not self._incoming[e.head]:
                self._incoming.pop(e.head, None)
        for tvid in list(e.tail):
            if eid in self._outgoing.get(tvid, set()):
                self._outgoing[tvid].remove(eid)
                if not self._outgoing[tvid]:
                    self._outgoing.pop(tvid, None)

    def prune(self, now_t: Timestamp, prune_threshold: float) -> int:
        """Remove edges with reliability < threshold; return count pruned."""
        to_remove: List[EdgeId] = [eid for eid, e in self._edges.items() if float(e.reliability) < prune_threshold]
        for eid in to_remove:
            self._remove_edge(eid)
        return len(to_remove)

    def snapshot(self) -> HypergraphSnapshot:
        # Shallow copies for safety
        vertices = dict(self._vertices)
        edges = dict(self._edges)
        incoming = {vid: set(eids) for vid, eids in self._incoming.items()}
        outgoing = {vid: set(eids) for vid, eids in self._outgoing.items()}
        summary = {
            "num_vertices": len(vertices),
            "num_edges": len(edges),
        }
        return HypergraphSnapshot(vertices=vertices, hyperedges=edges, incoming=incoming, outgoing=outgoing, summary=summary)


class StaticGraphConnectivity(GraphConnectivity):
    """
    Minimal connectivity oracle backed by a static mapping:
        postsyn neuron id -> iterable of presyn neuron ids
    """

    def __init__(self, mapping: Mapping[NeuronId, Iterable[NeuronId]]) -> None:
        self._map: Dict[NeuronId, Set[NeuronId]] = {k: set(v) for k, v in mapping.items()}

    def presyn_sources(self, neuron_id: NeuronId) -> Iterable[NeuronId]:
        return self._map.get(neuron_id, set())


__all__ = ["InMemoryHypergraph", "StaticGraphConnectivity"]