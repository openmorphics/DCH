# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Dynamic Hypergraph Construction (DHG) — TC-kNN candidate generation and admission.

Implements a default DHGConstructor that:
- Identifies presynaptic sources from a connectivity oracle
- Searches recent spikes within a temporal window (TC-kNN per presyn neuron)
- Generates unary and higher-order candidate hyperedges with temporal/refractory guards
- Deduplicates and enforces per-head candidate budgets
- Admits candidates into the hypergraph via HypergraphOps

References:
- Contracts: dch_core.interfaces (DHGConstructor, HypergraphOps, GraphConnectivity, etc.)
- Algorithm specs: docs/AlgorithmSpecs.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from dch_core.interfaces import (
    DHGConstructor,
    EdgeId,
    Event,
    GraphConnectivity,
    Hyperedge,
    HypergraphOps,
    NeuronId,
    Timestamp,
    Vertex,
    VertexId,
    Window,
    is_temporally_admissible,
    make_edge_id,
    make_vertex_id,
)


def _top_k_recent_spikes(
    hypergraph: HypergraphOps,
    neuron_id: NeuronId,
    window: Window,
    k: int,
) -> List[Vertex]:
    """
    Retrieve up to k most recent spikes for a given neuron within window.
    Fallback implementation uses window_query + filter; optimized backends may override via HypergraphOps specialization.
    """
    t0, t1 = window
    verts = [v for v in hypergraph.window_query(window) if v.neuron_id == neuron_id]
    verts.sort(key=lambda v: v.t, reverse=True)
    return verts[: max(0, k)]


def _cluster_by_time_proximity(
    vertices: Sequence[Vertex],
    delta_causal: int,
) -> List[List[Vertex]]:
    """
    Simple single-pass clustering by temporal proximity.
    Assumes vertices are sorted by time ascending.
    """
    if not vertices:
        return []
    cluster: List[Vertex] = [vertices[0]]
    clusters: List[List[Vertex]] = []
    for v in vertices[1:]:
        if v.t - cluster[-1].t <= delta_causal:
            cluster.append(v)
        else:
            clusters.append(cluster)
            cluster = [v]
    clusters.append(cluster)
    return clusters


def _refractory_ok(
    hypergraph: HypergraphOps,
    head_vertex: Vertex,
    refractory_rho: int,
) -> bool:
    """
    Enforce refractory constraint for the postsynaptic neuron: no other head spike for the same neuron
    should occur within (head.t - rho, head.t).
    """
    if refractory_rho <= 0:
        return True
    t0 = head_vertex.t - refractory_rho + 1
    t1 = head_vertex.t - 1
    if t1 < t0:
        return True
    window = (t0, t1)
    recent = hypergraph.window_query(window)
    return all(v.neuron_id != head_vertex.neuron_id for v in recent)


def _dedup_by_key(candidates: Sequence[Hyperedge]) -> List[Hyperedge]:
    """
    Deduplicate by (head, sorted(tail)) while keeping the first encountered edge.
    """
    seen: Set[Tuple[VertexId, Tuple[VertexId, ...]]] = set()
    out: List[Hyperedge] = []
    for e in candidates:
        key = (e.head, tuple(sorted(e.tail)))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _score_candidate(e: Hyperedge, vertex_times: Mapping[VertexId, Timestamp]) -> float:
    """
    Heuristic score for candidate selection under budget:
    - Prefer smaller tails (unary over higher-order)
    - Prefer recency (max tail time)
    """
    tail_size_penalty = 1.0 / (len(e.tail) ** 2)
    recency = max(vertex_times[vid] for vid in e.tail) if e.tail else 0
    # Normalize recency by absolute timestamp scale is not needed for ranking within a head
    return tail_size_penalty * (1.0 + 1e-9 * recency)


@dataclass
class DefaultDHGConstructor(DHGConstructor):
    """
    Default implementation of TC-kNN candidate generation and admission.
    """

    def generate_candidates_tc_knn(
        self,
        hypergraph: HypergraphOps,
        connectivity: GraphConnectivity,
        head_vertex: Vertex,
        window: Window,
        k: int,
        combination_order_max: int,
        causal_coincidence_delta: int,
        budget_per_head: int,
        init_reliability: float,
        refractory_rho: int,
    ) -> Sequence[Hyperedge]:
        # Derive (Δmin, Δmax) for candidate edges from the provided window.
        # window = [t_head - Δmax, t_head - Δmin] => Δmin = t_head - window[1], Δmax = t_head - window[0]
        t_head = head_vertex.t
        delta_min = max(0, t_head - window[1])
        delta_max = max(delta_min, t_head - window[0])

        presyn_set = list(connectivity.presyn_sources(head_vertex.neuron_id))

        # Collect recent spikes per presyn neuron
        recent_by_neuron: Mapping[NeuronId, List[Vertex]] = {
            n_i: _top_k_recent_spikes(hypergraph, n_i, window, k) for n_i in presyn_set
        }
        # Flatten and sort by time for grouping
        flat_recent: List[Vertex] = sorted(
            (v for lst in recent_by_neuron.values() for v in lst), key=lambda v: v.t
        )

        # Unary candidates
        candidates: List[Hyperedge] = []
        vtime: dict[VertexId, Timestamp] = {}

        for v in flat_recent:
            vtime[v.id] = v.t
            tail = {v.id}
            # Temporal admissibility (unary): v.t in [t_head - Δmax, t_head - Δmin]
            if not is_temporally_admissible([v.t], t_head, delta_min, delta_max):
                continue
            if not _refractory_ok(hypergraph, head_vertex, refractory_rho):
                continue
            e = Hyperedge(
                id=make_edge_id(head=head_vertex.id, tail=tail, t=t_head),
                tail=tail,
                head=head_vertex.id,
                delta_min=delta_min,
                delta_max=delta_max,
                refractory_rho=refractory_rho,
                reliability=init_reliability,
                provenance="unary",
            )
            candidates.append(e)

        # Higher-order candidates: cluster by time proximity then enumerate small combinations
        if combination_order_max >= 2 and causal_coincidence_delta > 0 and flat_recent:
            clusters = _cluster_by_time_proximity(flat_recent, causal_coincidence_delta)
            for group in clusters:
                if len(group) < 2:
                    continue
                # Enumerate combinations up to combination_order_max
                # For simplicity and tractability, only size-2 and size-3 if allowed
                group_sorted = sorted(group, key=lambda v: v.t)
                # size-2
                if combination_order_max >= 2:
                    for i in range(len(group_sorted)):
                        for j in range(i + 1, len(group_sorted)):
                            t_tail = [group_sorted[i].t, group_sorted[j].t]
                            if not is_temporally_admissible(t_tail, t_head, delta_min, delta_max):
                                continue
                            if not _refractory_ok(hypergraph, head_vertex, refractory_rho):
                                continue
                            tail_ids: Set[VertexId] = {group_sorted[i].id, group_sorted[j].id}
                            for vid in tail_ids:
                                vtime[vid] = hypergraph.get_vertex(vid).t if hypergraph.get_vertex(vid) else 0
                            e2 = Hyperedge(
                                id=make_edge_id(head=head_vertex.id, tail=tail_ids, t=t_head),
                                tail=tail_ids,
                                head=head_vertex.id,
                                delta_min=delta_min,
                                delta_max=delta_max,
                                refractory_rho=refractory_rho,
                                reliability=init_reliability,
                                provenance="pair",
                            )
                            candidates.append(e2)
                # size-3
                if combination_order_max >= 3 and len(group_sorted) >= 3:
                    for i in range(len(group_sorted)):
                        for j in range(i + 1, len(group_sorted)):
                            for k3 in range(j + 1, len(group_sorted)):
                                t_tail = [group_sorted[i].t, group_sorted[j].t, group_sorted[k3].t]
                                if not is_temporally_admissible(t_tail, t_head, delta_min, delta_max):
                                    continue
                                if not _refractory_ok(hypergraph, head_vertex, refractory_rho):
                                    continue
                                tail_ids = {group_sorted[i].id, group_sorted[j].id, group_sorted[k3].id}
                                for vid in tail_ids:
                                    v = hypergraph.get_vertex(vid)
                                    vtime[vid] = v.t if v else 0
                                e3 = Hyperedge(
                                    id=make_edge_id(head=head_vertex.id, tail=tail_ids, t=t_head),
                                    tail=tail_ids,
                                    head=head_vertex.id,
                                    delta_min=delta_min,
                                    delta_max=delta_max,
                                    refractory_rho=refractory_rho,
                                    reliability=init_reliability,
                                    provenance="triple",
                                )
                                candidates.append(e3)

        # Deduplicate
        candidates = _dedup_by_key(candidates)

        # Enforce per-head budget using heuristic scoring
        if budget_per_head > 0 and len(candidates) > budget_per_head:
            ranked = sorted(
                candidates,
                key=lambda e: _score_candidate(e, vtime),
                reverse=True,
            )
            candidates = ranked[:budget_per_head]

        return candidates

    def admit(
        self,
        hypergraph: HypergraphOps,
        candidates: Sequence[Hyperedge],
    ) -> Sequence[EdgeId]:
        """
        Insert candidates into the hypergraph with dedup and budgets handled by HypergraphOps.
        """
        if not candidates:
            return []
        return list(hypergraph.insert_hyperedges(candidates))