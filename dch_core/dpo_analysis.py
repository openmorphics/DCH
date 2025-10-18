# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
DPO analysis utilities: small, deterministic helpers for confluence and termination checks.

Scope
- analysis-only, stdlib-only
- non-invasive: reuses existing DPO primitives; does not alter engine behavior
- tiny snapshots built in-memory; tests provide seed edges explicitly

Exports
- analyze_critical_pairs
- check_termination
- snapshot_edges
- clone_adapter_with
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from dch_core.interfaces import EdgeId, VertexId, Hyperedge, Vertex, make_edge_id
from dch_core.hypergraph_mem import InMemoryHypergraph
from dch_core.dpo import (
    DPO_Rule,
    DPO_Match,
    DPO_ApplyResult,
    DPOGraphAdapter,
    DPOEngine,
)

# Reserved keys for passing seed context through matches_by_rule without changing function signature
_SEED_EDGES_KEY = "__seed_edges__"         # List[Tuple[List[VertexIdLike], VertexIdLike, Dict[str, Any]]]
_EXTRA_VERTICES_KEY = "__extra_vertices__" # List[VertexIdLike]


# -------------------------
# Helpers
# -------------------------

def _to_vid(x: Any) -> VertexId:
    return VertexId(str(x))


def _ensure_vertices_exist(hg: InMemoryHypergraph, vids: Iterable[VertexId]) -> None:
    # Directly populate the in-memory backend to respect the exact VertexId strings provided by callers/tests.
    for vid in vids:
        if hg.get_vertex(vid) is not None:
            continue
        s = str(vid)
        neuron_id = 0
        t = 0
        # Best-effort parse if format resembles "neuron@time"
        if "@" in s:
            try:
                a, b = s.split("@", 1)
                neuron_id = int(a)
                t = int(b)
            except Exception:
                neuron_id = 0
                t = 0
        else:
            # If plain integer-like, map it to both neuron_id and t; otherwise defaults remain 0.
            try:
                v = int(s)
                neuron_id = v
                t = v
            except Exception:
                pass
        vtx = Vertex(id=vid, neuron_id=neuron_id, t=t)
        # Insert into internal structures (analysis-only utility; safe for in-memory backend)
        hg._vertices[vid] = vtx  # type: ignore[attr-defined]
        hg._time_index.append(vtx)  # type: ignore[attr-defined]


def snapshot_edges(adapter: DPOGraphAdapter) -> set[tuple[tuple[str, ...], str, bool]]:
    """
    Produce a canonical set representation of current edges:
        (sorted_tail_vertex_ids_tuple, head_vertex_id, frozen_flag)

    - Vertex identifiers are represented as strings for determinism.
    - 'frozen' is read from edge.attributes.get("frozen", False).
    """
    snap = adapter.hg.snapshot()
    result: set[tuple[tuple[str, ...], str, bool]] = set()
    for e in snap.hyperedges.values():
        tails = tuple(sorted(str(v) for v in e.tail))
        head = str(e.head)
        frozen = bool((e.attributes or {}).get("frozen", False))
        result.add((tails, head, frozen))
    return result


def clone_adapter_with(seed_edges: List[Tuple[Sequence[Any], Any, Mapping[str, Any]]]) -> DPOGraphAdapter:
    """
    Build a fresh DPOGraphAdapter with provided seed edges and attributes.

    Each seed is a tuple: (tails, head, attrs)
    - tails: Sequence of vertex identifiers (any type convertible to VertexId via str)
    - head: Vertex identifier (convertible to VertexId via str)
    - attrs: Dict with optional keys:
        - id or edge_id: explicit EdgeId to use
        - reliability: float
        - counts_success: int
        - counts_miss: int
        - frozen: bool
        - delta_min: int (default 0)
        - delta_max: int (default 0)
        - refractory_rho: int (default 0)
        - t: optional integer to use in make_edge_id if id not provided (default 0)
        - attributes: dict merged into Hyperedge.attributes (alongside frozen)

    Notes:
    - Vertices referenced by tails/head are created directly with the exact VertexId strings.
    - The in-memory backend is used; this helper relies on its internal structures for insertion.
    """
    hg = InMemoryHypergraph()

    # Materialize all vertices first
    all_vids: set[VertexId] = set()
    for tails, head, _attrs in seed_edges:
        all_vids.update(_to_vid(v) for v in tails)
        all_vids.add(_to_vid(head))
    _ensure_vertices_exist(hg, all_vids)

    # Build and insert edges
    candidates: List[Hyperedge] = []
    for tails, head, _attrs in seed_edges:
        tail_set = { _to_vid(v) for v in tails }
        head_vid = _to_vid(head)
        attrs = dict(_attrs or {})

        # Attributes and top-level fields
        reliability = float(attrs.pop("reliability", 0.10))
        counts_success = int(attrs.pop("counts_success", 0))
        counts_miss = int(attrs.pop("counts_miss", 0))
        frozen_flag = bool(attrs.pop("frozen", False))
        refractory_rho = int(attrs.pop("refractory_rho", 0))
        delta_min = int(attrs.pop("delta_min", 0))
        delta_max = int(attrs.pop("delta_max", 0))
        t_nonce = int(attrs.pop("t", 0))
        explicit_id = attrs.pop("id", attrs.pop("edge_id", None))

        edge_id: EdgeId
        if explicit_id is not None:
            edge_id = EdgeId(str(explicit_id))
        else:
            edge_id = make_edge_id(head=head_vid, tail=tail_set, t=t_nonce)

        edge_attrs = dict(attrs.pop("attributes", {}))
        edge_attrs["frozen"] = frozen_flag
        # Merge any leftover keys as attributes for transparency
        for k, v in list(attrs.items()):
            edge_attrs.setdefault(k, v)

        he = Hyperedge(
            id=edge_id,
            tail=tail_set,
            head=head_vid,
            delta_min=delta_min,
            delta_max=delta_max,
            refractory_rho=refractory_rho,
            reliability=reliability,
            counts_success=counts_success,
            counts_miss=counts_miss,
            provenance="analysis:seed",
            attributes=edge_attrs,
        )
        candidates.append(he)

    admitted = list(hg.insert_hyperedges(candidates))
    # Ensure all seeds were admitted (dedup can drop if same key added twice)
    # For deterministic tests we don't error; analysis functions don't require strict admission.
    return DPOGraphAdapter(hg)


# -------------------------
# Critical-pair analysis
# -------------------------

def _collect_required_vertices(
    A: DPO_Rule, mA: DPO_Match, B: DPO_Rule, mB: DPO_Match
) -> set[VertexId]:
    req: set[VertexId] = set()
    # From matches (explicit vertices for GROW precondition checks)
    for v in (mA.vertices or []):
        req.add(_to_vid(v))
    for v in (mB.vertices or []):
        req.add(_to_vid(v))
    # From rule params (if present)
    for rule in (A, B):
        tails_param = rule.params.get("tails", [])
        head_param = rule.params.get("head", None)
        for v in tails_param or []:
            req.add(_to_vid(v))
        if head_param is not None:
            req.add(_to_vid(head_param))
    return req


def _apply_sequence(
    eng: DPOEngine,
    seq: Sequence[Tuple[DPO_Rule, DPO_Match]],
    adapter: DPOGraphAdapter,
) -> Tuple[set[tuple[tuple[str, ...], str, bool]], int, str]:
    reasons: List[str] = []
    for rule, match in seq:
        res = eng.apply(rule, match, adapter)
        tag = "ok" if res.applied else "fail"
        reasons.append(f"{rule.name}:{tag}:{res.reason}")
    snap_set = snapshot_edges(adapter)
    num_edges = len(adapter.hg.snapshot().hyperedges)
    return snap_set, num_edges, " | ".join(reasons)


def analyze_critical_pairs(
    rules: List[DPO_Rule],
    matches_by_rule: Dict[str, List[DPO_Match]],
    *,
    max_pairs: int = 100,
    max_steps: int = 64,
) -> Dict[str, Any]:
    """
    For each ordered pair of rules (A,B) and available matches, simulate A→B and B→A
    starting from fresh identical adapter snapshots and compare end states.

    Seeding:
    - If matches_by_rule contains key '__seed_edges__', it is treated as the list of seed edges
      accepted by clone_adapter_with(...) to build initial identical graphs.
    - If '__extra_vertices__' is provided, those VertexIds are ensured to exist before application.

    Equality:
    - Uses snapshot_edges(...) to compare canonical edge triples plus edge counts.

    Returns a report dict:
    {
      "total_pairs": int,
      "joinable": int,
      "divergent": int,
      "pairs": [
        {"A": A.name, "B": B.name, "joinable": bool, "reason": str}
      ]
    }
    """
    # Pull seeds and extras if provided
    seed_edges: List[Tuple[Sequence[Any], Any, Mapping[str, Any]]] = list(matches_by_rule.get(_SEED_EDGES_KEY, []))  # type: ignore[assignment]
    extra_vertices: List[Any] = list(matches_by_rule.get(_EXTRA_VERTICES_KEY, []))  # type: ignore[assignment]

    # Build a reusable base snapshot serializer (deterministic rebuild for each trial)
    def _fresh_adapter() -> DPOGraphAdapter:
        adapter = clone_adapter_with(seed_edges)
        if extra_vertices:
            _ensure_vertices_exist(adapter.hg, (_to_vid(v) for v in extra_vertices))
        return adapter

    eng = DPOEngine()
    pairs: List[Dict[str, Any]] = []
    total = 0
    joined = 0

    # Map rule name -> matches list
    mb: Dict[str, List[DPO_Match]] = {str(k): list(v) for k, v in matches_by_rule.items() if k not in (_SEED_EDGES_KEY, _EXTRA_VERTICES_KEY)}

    for i, A in enumerate(rules):
        mAs = mb.get(A.name, [])
        if not mAs:
            continue
        for j, B in enumerate(rules):
            mBs = mb.get(B.name, [])
            if not mBs:
                continue
            for mA in mAs:
                for mB in mBs:
                    if total >= int(max_pairs):
                        break

                    # Fresh identical snapshots for AB and BA
                    g_ab = _fresh_adapter()
                    g_ba = _fresh_adapter()

                    # Ensure any required vertices (e.g., for GROW) exist in both adapters
                    req_vids = _collect_required_vertices(A, mA, B, mB)
                    if req_vids:
                        _ensure_vertices_exist(g_ab.hg, req_vids)
                        _ensure_vertices_exist(g_ba.hg, req_vids)

                    # Apply A then B
                    s_ab, n_ab, r_ab = _apply_sequence(eng, [(A, mA), (B, mB)], g_ab)
                    # Apply B then A
                    s_ba, n_ba, r_ba = _apply_sequence(eng, [(B, mB), (A, mA)], g_ba)

                    joinable = (s_ab == s_ba) and (n_ab == n_ba)
                    if joinable:
                        joined += 1
                        reason = f"joinable; AB[{r_ab}] == BA[{r_ba}]"
                    else:
                        reason = f"divergent; AB[{r_ab}] vs BA[{r_ba}]"

                    pairs.append({"A": A.name, "B": B.name, "joinable": bool(joinable), "reason": reason})
                    total += 1
                if total >= int(max_pairs):
                    break
            if total >= int(max_pairs):
                break
        if total >= int(max_pairs):
            break

    return {
        "total_pairs": total,
        "joinable": joined,
        "divergent": max(0, total - joined),
        "pairs": pairs,
    }


# -------------------------
# Bounded termination check
# -------------------------

def check_termination(
    rules: List[DPO_Rule],
    *,
    max_grow: Optional[int] = None,
    max_steps: int = 256,
) -> Dict[str, Any]:
    """
    Provide a bounded termination analysis using simple structural criteria:

    - If rules contain no "GROW" kinds → terminate=True (only PRUNE/FREEZE which are non-expansive).
    - If rules contain "GROW" but max_grow is not None and ≥ 0 → terminate=True with budgeted bound.
    - Else return terminate="unknown" with reason indicating unbounded growth potential.

    Returns:
      {
        "terminate": bool | "unknown",
        "reason": str,
        "constraints": {"max_grow": max_grow, "max_steps": max_steps, "kinds": sorted_unique_kinds}
      }
    """
    kinds = sorted({str(r.kind).upper() for r in rules})
    has_grow = any(str(r.kind).upper() == "GROW" for r in rules)

    if not has_grow:
        return {
            "terminate": True,
            "reason": "No GROW rules present; PRUNE/FREEZE are non-expansive under this model.",
            "constraints": {"max_grow": max_grow, "max_steps": int(max_steps), "kinds": kinds},
        }

    if max_grow is not None and int(max_grow) >= 0:
        return {
            "terminate": True,
            "reason": "GROW is present but bounded by explicit max_grow budget.",
            "constraints": {"max_grow": int(max_grow), "max_steps": int(max_steps), "kinds": kinds},
        }

    return {
        "terminate": "unknown",
        "reason": "GROW present without budget; termination cannot be guaranteed under this simple analysis.",
        "constraints": {"max_grow": max_grow, "max_steps": int(max_steps), "kinds": kinds},
    }


__all__ = ["analyze_critical_pairs", "check_termination", "snapshot_edges", "clone_adapter_with"]