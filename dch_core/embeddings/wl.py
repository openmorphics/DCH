# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Weisfeiler–Lehman (WL) style hyperpath embedding for DCH (torch-free).

This module provides:
- WLParams: configuration (with backward-compatible aliases d/iters/salt)
- WLHyperpathEmbedding: deterministic embedding + canonical labeling

Key properties:
- Deterministic across runs given the same params and inputs
- Invariant to tail permutation within a hyperedge
- Invariant to absolute time shifts (uses relative delays Δt)
- Torchless; depends only on Python stdlib and numpy

Backward compatibility:
- Supports the existing tests that use WLParams(d=..., iters=..., salt=...)
- If EdgeId/VertexId strings are not in canonical form, falls back to a
  simple token hashing scheme over HEAD/EDGE/LENGTH while remaining deterministic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from dch_core.interfaces import EdgeId, Hyperedge, Hyperpath, VertexId


@dataclass
class WLParams:
    """Parameters for the WL-style hyperpath embedding."""
    vector_dim: int = 256
    iterations: int = 2
    normalize: bool = True
    hash_seed: int = 0
    include_reliability: bool = False
    time_resolution: float = 1.0
    # Backward-compat aliases (used by older tests)
    d: Optional[int] = None
    iters: Optional[int] = None
    salt: Optional[int] = None

    def vec_dim(self) -> int:
        return int(self.d if self.d is not None else self.vector_dim)

    def iters_count(self) -> int:
        return int(self.iters if self.iters is not None else self.iterations)

    def seed(self) -> int:
        return int(self.salt if self.salt is not None else self.hash_seed)


class WLHyperpathEmbedding:
    """
    WL-inspired, deterministic embedding for DCH Hyperpaths.

    - canonical_label(hyperpath): text label stable to tail permutations and time shifts
    - embed(hyperpath): numpy.ndarray shape (vector_dim,)
    - embed_batch(hyperpaths) / batch_embed(hyperpaths): stacked embeddings

    Optionally accepts an edge resolver to read Hyperedge.reliability when
    include_reliability=True. If not provided, reliability defaults to 1.0.
    """

    def __init__(
        self,
        params: WLParams,
        edge_lookup: Optional[Mapping[EdgeId, Hyperedge]] = None,
        edge_resolver: Optional[Callable[[EdgeId], Optional[Hyperedge]]] = None,
    ) -> None:
        self.params = params
        self._edge_lookup = edge_lookup
        self._edge_resolver = edge_resolver
        seed_int = self.params.seed()
        self._seed_bytes = int(seed_int).to_bytes(8, byteorder="little", signed=False)

    # -----------------------
    # Public API
    # -----------------------

    def canonical_label(self, hyperpath: Hyperpath) -> str:
        """
        Deterministic canonical string using only neuron ids and Δt (rounded).
        Format: "HEAD:{sink}|EDGES:[(head, ((n,dt),...), [rel]) , ...]"
        """
        sink_neuron, _ = self._parse_vertex_id(hyperpath.head)
        edge_tokens: List[str] = []

        for eid in hyperpath.edges:
            parsed = self._parse_edge_id(eid)
            if parsed is None:
                # Fallback token for non-canonical id
                edge_tokens.append(f"RAW:{str(eid)}")
                continue
            head_vid, tail_vids = parsed
            h_neu, h_t = self._parse_vertex_id(head_vid)
            tails_pairs: List[Tuple[str, int]] = []
            for tv in tail_vids:
                t_neu, t_t = self._parse_vertex_id(tv)
                dt = self._delta_t(h_t, t_t)
                tails_pairs.append((str(t_neu), int(dt)))
            tails_pairs.sort()

            tup: Tuple[object, ...]
            if self.params.include_reliability:
                rel = self._edge_reliability(eid)
                rel_r = round(rel, 3)
                tup = (str(h_neu), tuple(tails_pairs), f"{rel_r:.3f}")
            else:
                tup = (str(h_neu), tuple(tails_pairs))
            edge_tokens.append(repr(tup))

        edge_tokens.sort()
        return f"HEAD:{sink_neuron}|EDGES:[{','.join(edge_tokens)}]"

    def embed(self, hyperpath: Hyperpath) -> np.ndarray:
        """
        Compute WL-style embedding for a single hyperpath.
        Returns a numpy vector of length vector_dim (float64).
        """
        d = self.params.vec_dim()
        if d <= 0:
            raise ValueError("vector_dim must be positive")

        # Empty hyperpath => zero vector by convention
        if not hyperpath.edges:
            return np.zeros(d, dtype=np.float64)

        edges_info = self._collect_edges_info(hyperpath.edges)
        if not edges_info:
            # Fallback: token hashing over raw identifiers (deterministic + normalized)
            vec = np.zeros(d, dtype=np.float64)
            tokens: List[str] = [f"HEAD:{hyperpath.head}", f"L:{int(hyperpath.length)}"]
            for e in sorted(hyperpath.edges):
                tokens.append(f"E:{e}")
            for tok in tokens:
                idx, sgn = self._hash_index_sign(tok, d)
                vec[idx] += float(sgn)
            if self.params.normalize:
                n = float(np.linalg.norm(vec))
                if n > 0.0:
                    vec /= n
            return vec

        # WL refinement on vertex labels
        vlabels: Dict[VertexId, str] = self._initial_vertex_labels(edges_info, hyperpath.head)
        iters = self.params.iters_count()
        for _ in range(max(0, iters)):
            contexts: Dict[VertexId, List[str]] = {v: [] for v in vlabels.keys()}
            for (h_vid, h_neu, h_t, tails) in edges_info:
                h_lab = vlabels[h_vid]
                tail_labs = [vlabels[tv] for (tv, _neu, _tt, _dt) in tails]
                dts_str = [str(dt) for (_tv, _n, _tt, dt) in tails]
                head_ctx = f"H:{h_lab}|T:{repr(sorted(tail_labs))}|D:{repr(sorted(dts_str))}"
                contexts[h_vid].append(head_ctx)
                # Tail-side contexts (optional but improves discrimination)
                for (tv, _n, _tt, dt) in tails:
                    t_lab = vlabels[tv]
                    tail_ctx = f"T:{t_lab}|H:{h_lab}|D:{dt}"
                    contexts[tv].append(tail_ctx)
            # Update labels using hashed multiset of contexts
            new_labels: Dict[VertexId, str] = {}
            for v, lab in vlabels.items():
                ctxs = contexts.get(v, [])
                if ctxs:
                    ctxs_join = "||".join(sorted(ctxs))
                    new_lab = self._hash_label(lab + "⟂" + self._hash_label(ctxs_join))
                else:
                    new_lab = self._hash_label(lab + "⟂")
                new_labels[v] = new_lab
            vlabels = new_labels

        # Feature hashing of refined edge labels
        vec = np.zeros(d, dtype=np.float64)
        for (h_vid, _h_neu, _h_t, tails) in edges_info:
            h_lab = vlabels[h_vid]
            tail_pairs = sorted([(vlabels[tv], str(dt)) for (tv, _n, _tt, dt) in tails])
            label = f"E:{h_lab}|T:{repr(tail_pairs)}"
            weight = 1.0
            if self.params.include_reliability:
                rel = self._edge_reliability(self._compose_edge_id(h_vid, [tv for (tv, _n, _tt, _dt) in tails]))
                weight = round(rel, 3)
                label = f"{label}|R:{weight:.3f}"
            idx, sgn = self._hash_index_sign(label, d)
            vec[idx] += float(sgn) * float(weight)

        if self.params.normalize:
            n = float(np.linalg.norm(vec))
            if n > 0.0:
                vec /= n
        return vec

    def batch_embed(self, hyperpaths: Sequence[Hyperpath]) -> np.ndarray:
        """Compatibility with EmbeddingEngine Protocol: batch embeddings."""
        d = self.params.vec_dim()
        out = np.zeros((len(hyperpaths), d), dtype=np.float64)
        for i, hp in enumerate(hyperpaths):
            out[i, :] = self.embed(hp)
        return out

    # Alias per technical spec
    def embed_batch(self, hyperpaths: Sequence[Hyperpath]) -> np.ndarray:
        return self.batch_embed(hyperpaths)

    # -----------------------
    # Internals
    # -----------------------

    def _initial_vertex_labels(self, edges_info: List[Tuple[VertexId, str, Optional[int], List[Tuple[VertexId, str, Optional[int], int]]]], sink: VertexId) -> Dict[VertexId, str]:
        """Initialize vertex labels as neuron_id strings, or fallback to raw id."""
        labels: Dict[VertexId, str] = {}
        # include all vertices in edges
        for (h_vid, h_neu, _h_t, tails) in edges_info:
            labels.setdefault(h_vid, str(h_neu))
            for (tv, t_neu, _tt, _dt) in tails:
                labels.setdefault(tv, str(t_neu))
        # ensure sink is present
        s_neu, _ = self._parse_vertex_id(sink)
        labels.setdefault(sink, str(s_neu))
        # Fallback for any unresolved (non-canonical ids)
        for v, lab in list(labels.items()):
            if lab is None or lab == "":
                labels[v] = f"VID:{v}"
        return labels

    def _edge_reliability(self, eid: EdgeId) -> float:
        """Resolve Hyperedge.reliability or default to 1.0."""
        he: Optional[Hyperedge] = None
        if self._edge_resolver is not None:
            he = self._edge_resolver(eid)  # type: ignore[arg-type]
        if he is None and self._edge_lookup is not None:
            he = self._edge_lookup.get(eid)  # type: ignore[index]
        if he is None:
            return 1.0
        try:
            r = float(he.reliability)
        except Exception:
            r = 1.0
        # clamp to [0,1]
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        return r

    def _hash_index_sign(self, s: str, d: int) -> Tuple[int, int]:
        """Feature hashing: index in [0,d), sign in {-1,+1}."""
        msg = s.encode("utf-8")
        h_idx = hashlib.blake2b(msg, digest_size=8, key=self._seed_bytes + b"IDX").digest()
        h_sgn = hashlib.blake2b(msg, digest_size=1, key=self._seed_bytes + b"SGN").digest()
        idx = int.from_bytes(h_idx, byteorder="little", signed=False) % max(1, d)
        sgn = 1 if (h_sgn[0] & 1) else -1
        return idx, sgn

    def _hash_label(self, s: str) -> str:
        """Hashed label string used for WL refinements."""
        return hashlib.blake2b(s.encode("utf-8"), digest_size=16, key=self._seed_bytes + b"LBL").hexdigest()

    def _delta_t(self, head_time: Optional[int], tail_time: Optional[int]) -> int:
        if head_time is None or tail_time is None:
            return 0
        res = float(self.params.time_resolution if self.params.time_resolution else 1.0)
        return int(round((int(head_time) - int(tail_time)) / res))

    def _parse_vertex_id(self, vid: VertexId) -> Tuple[str, Optional[int]]:
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
        s = str(eid)
        # Strip nonce after '#'
        left, _sep_hash, _nonce = s.partition("#")
        # Split head and tails
        head_s, sep, tails_s = left.partition("&")
        if sep == "":
            return None
        tails = [VertexId(x) for x in tails_s.split(",") if x]
        return VertexId(head_s), tails

    def _collect_edges_info(
        self, eids: Sequence[EdgeId]
    ) -> List[Tuple[VertexId, str, Optional[int], List[Tuple[VertexId, str, Optional[int], int]]]]:
        """
        Parse edge ids into structured info:
        List of (head_vid, head_neuron, head_time, tails=[(tail_vid, tail_neuron, tail_time, Δt), ...])
        """
        info: List[Tuple[VertexId, str, Optional[int], List[Tuple[VertexId, str, Optional[int], int]]]] = []
        for eid in eids:
            parsed = self._parse_edge_id(eid)
            if parsed is None:
                continue
            h_vid, tail_vids = parsed
            h_neu, h_t = self._parse_vertex_id(h_vid)
            tails_list: List[Tuple[VertexId, str, Optional[int], int]] = []
            for tv in tail_vids:
                t_neu, t_t = self._parse_vertex_id(tv)
                dt = self._delta_t(h_t, t_t)
                tails_list.append((tv, t_neu, t_t, dt))
            # Sort tails for determinism
            tails_list.sort(key=lambda x: (str(x[1]), int(x[3])))
            info.append((h_vid, h_neu, h_t, tails_list))
        return info

    def _compose_edge_id(self, head: VertexId, tails: Sequence[VertexId]) -> EdgeId:
        """
        Compose a canonical-like EdgeId string for lookup purposes (ignores nonce).
        """
        tails_s = ",".join(sorted([str(t) for t in tails]))
        return EdgeId(f"{head}&{tails_s}#0")


__all__ = ["WLParams", "WLHyperpathEmbedding"]