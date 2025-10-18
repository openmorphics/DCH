# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Minimal Double Pushout (DPO) rewrite prototype for structural plasticity operations.

Scope
- This prototype provides a very small, auditable DPO-style engine to apply:
  * GROW: add a new hyperedge
  * PRUNE: remove a hyperedge under a reliability threshold
  * FREEZE: mark a hyperedge as frozen (attribute toggle)

Design notes
- We model rules as (L, K, R) schematics (placeholders) with typed metadata sufficient for matching.
- Matching is explicit and passed in via DPO_Match. For this prototype we match on concrete ids only.
- Application is deterministic and limited to adapter calls (no hidden side effects).
- The adapter is intentionally thin around the in-memory hypergraph backend.

Integration
- This module integrates via DPOGraphAdapter and does not change any core protocols (see dch_core.interfaces).
- Reliability checks use either the stored reliability or a best-effort Beta posterior mean from counters.

Caveats
- Temporal constraints, budgets, and deduplication policies from DHG are not replicated here.
- GROW computes simple default temporal parameters when constructing a new Hyperedge.
- Removal uses the in-memory backend's internal _remove_edge when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Literal

from dch_core.interfaces import (
    EdgeId,
    VertexId,
    Hyperedge,
    Timestamp,
    make_edge_id,
)
from dch_core.hypergraph_mem import InMemoryHypergraph


# -------------------------
# Data structures
# -------------------------

@dataclass
class DPO_LKR:
    """Container for the (L, K, R) schematic with minimal symbolic metadata."""
    L: Mapping[str, Any] = field(default_factory=dict)
    K: Mapping[str, Any] = field(default_factory=dict)
    R: Mapping[str, Any] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class DPO_Rule:
    name: str
    kind: Literal["GROW", "PRUNE", "FREEZE"]
    lkr: DPO_LKR
    preconditions: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class DPO_Match:
    """Concrete binding for rule application."""
    vertices: Sequence[VertexId] = field(default_factory=list)  # concrete vertices bound to L and/or K
    edge_id: Optional[EdgeId] = None                            # when operating on a specific edge


@dataclass
class DPO_ApplyResult:
    applied: bool
    reason: str
    changes: Mapping[str, Any] = field(default_factory=dict)  # e.g., {"added_edges":[...], "removed_edges":[...], "state_changes":[...]}


# -------------------------
# Graph adapter
# -------------------------

class DPOGraphAdapter:
    """
    Thin adapter around InMemoryHypergraph for DPO operations.
    This adapter intentionally uses only public APIs when possible,
    with a best-effort fallback to internal removal for single-edge deletes.
    """

    def __init__(self, hypergraph: InMemoryHypergraph) -> None:
        self.hg = hypergraph

    # ---- Accessors ----

    def get_vertex(self, v_id: VertexId) -> Optional[Any]:
        try:
            return self.hg.get_vertex(VertexId(str(v_id)))
        except Exception:
            return None

    def get_edge(self, e_id: EdgeId) -> Optional[Hyperedge]:
        try:
            return self.hg.get_edge(EdgeId(str(e_id)))
        except Exception:
            return None

    # ---- Mutators ----

    def _find_existing_edge_by_head_tail(self, head: VertexId, tail_set: Sequence[VertexId]) -> Optional[EdgeId]:
        """Best-effort duplicate check by scanning incoming edges of head."""
        try:
            inc = self.hg.get_incoming_edges(VertexId(str(head)))
            tail_sorted = tuple(sorted(VertexId(str(v)) for v in tail_set))
            for eid in inc:
                e = self.hg.get_edge(eid)
                if e is None:
                    continue
                if tuple(sorted(e.tail)) == tail_sorted:
                    return eid
            return None
        except Exception:
            return None

    def add_edge(self, tails: Sequence[VertexId], head: VertexId, attrs: Optional[Mapping[str, Any]] = None) -> Tuple[Optional[EdgeId], bool]:
        """
        Insert a new unary/higher-order edge with simple default temporal parameters.

        Returns (edge_id, created)
        - If a duplicate is detected, returns (existing_id, False).
        - On insert success, returns (new_id, True).
        - On failure, returns (None, False).
        """
        attrs = dict(attrs or {})
        frozen_flag = bool(attrs.get("frozen", False))

        # Pre-check for duplicates
        dupe = self._find_existing_edge_by_head_tail(head, tails)
        if dupe is not None:
            return EdgeId(str(dupe)), False

        # Resolve timestamps for a simple delta window
        head_v = self.get_vertex(head)
        tail_vs = [self.get_vertex(vid) for vid in tails]
        if head_v is None or not all(tv is not None for tv in tail_vs):
            # Cannot construct without vertices
            return None, False

        t_head: Timestamp = int(getattr(head_v, "t", 0))
        tail_times: List[int] = [int(getattr(tv, "t", t_head)) for tv in tail_vs]
        # Simple admissibility window: [0, max_delay] bound by observed times
        max_delay = max(0, max((t_head - ti) for ti in tail_times) if tail_times else 0)

        tail_set = {VertexId(str(vid)) for vid in tails}
        eid = make_edge_id(head=VertexId(str(head)), tail=tail_set, t=t_head)

        he = Hyperedge(
            id=eid,
            tail=tail_set,
            head=VertexId(str(head)),
            delta_min=0,
            delta_max=int(max_delay),
            refractory_rho=0,
            reliability=0.10,
            provenance="dpo:grow",
            attributes={"frozen": frozen_flag, **{k: v for k, v in attrs.items() if k != "frozen"}},
        )

        admitted = list(self.hg.insert_hyperedges([he]))
        if not admitted:
            # Could be a race or dedup; try to locate existing
            existing = self._find_existing_edge_by_head_tail(head, tails)
            return (EdgeId(str(existing)) if existing is not None else None), False
        return EdgeId(str(admitted[0])), True

    def remove_edge(self, e_id: EdgeId) -> bool:
        """Best-effort single-edge removal for the in-memory backend."""
        try:
            e = self.get_edge(e_id)
            if e is None:
                return False
            # Prefer internal removal if available
            if hasattr(self.hg, "_remove_edge"):
                self.hg._remove_edge(EdgeId(str(e_id)))  # type: ignore[attr-defined]
                return True
            # Fallback: no general single-edge removal in the interface
            return False
        except Exception:
            return False

    def set_edge_attr(self, e_id: EdgeId, key: str, value: Any) -> None:
        e = self.get_edge(e_id)
        if e is None:
            raise KeyError(f"Edge not found: {e_id}")
        try:
            e.attributes[key] = value
        except Exception:
            # Ensure attributes exists
            try:
                e.attributes = dict(e.attributes or {})
            except Exception:
                e.attributes = {}
            e.attributes[key] = value

    def get_reliability(self, e_id: EdgeId) -> float:
        """Posterior mean from counters if available; fallback to stored reliability."""
        e = self.get_edge(e_id)
        if e is None:
            return 0.0
        try:
            s = float(getattr(e, "counts_success", 0.0))
            f = float(getattr(e, "counts_miss", 0.0))
            # Beta(1,1) prior posterior mean
            num = 1.0 + s
            den = 2.0 + s + f
            if den > 0.0:
                return float(num / den)
        except Exception:
            pass
        try:
            return float(getattr(e, "reliability", 0.0))
        except Exception:
            return 0.0


# -------------------------
# DPO engine
# -------------------------

class DPOEngine:
    """
    Minimal DPO engine applying rules over the adapter. Deterministic, no globals.
    """

    def __init__(self, *, theta_prune: float = 0.2, theta_freeze: float = 0.95) -> None:
        self.theta_prune = float(theta_prune)
        self.theta_freeze = float(theta_freeze)

    def apply(self, rule: DPO_Rule, match: DPO_Match, g: DPOGraphAdapter) -> DPO_ApplyResult:
        kind = str(rule.kind).upper()

        if kind == "GROW":
            # Params provide tails/head; precondition: vertices exist
            tails_raw = rule.params.get("tails", [])
            head_raw = rule.params.get("head")
            attrs = dict(rule.params.get("attributes", {}))

            tails: List[VertexId] = [VertexId(str(v)) for v in tails_raw]
            head: VertexId = VertexId(str(head_raw)) if head_raw is not None else VertexId("")

            # Precondition: all vertices exist
            verts_to_check = list(match.vertices) or (tails + ([head] if head else []))
            if not verts_to_check or any(g.get_vertex(v) is None for v in verts_to_check):
                return DPO_ApplyResult(applied=False, reason="Precondition failed: vertices missing", changes={})

            # Check duplicate before insert
            existing = g._find_existing_edge_by_head_tail(head, tails)
            if existing is not None:
                return DPO_ApplyResult(applied=False, reason="Duplicate edge", changes={})

            eid, created = g.add_edge(tails=tails, head=head, attrs={"frozen": False, **attrs})
            if created and eid is not None:
                return DPO_ApplyResult(applied=True, reason="Edge added", changes={"added_edges": [str(eid)]})
            elif eid is not None:
                return DPO_ApplyResult(applied=False, reason="Edge existed", changes={})
            else:
                return DPO_ApplyResult(applied=False, reason="Insert failed", changes={})

        elif kind == "PRUNE":
            # Precondition: edge exists and reliability <= theta
            eid_raw = rule.params.get("edge_id", match.edge_id)
            if eid_raw is None:
                return DPO_ApplyResult(applied=False, reason="Missing edge_id", changes={})
            eid = EdgeId(str(eid_raw))
            if g.get_edge(eid) is None:
                return DPO_ApplyResult(applied=False, reason="Edge not found", changes={})
            theta = float(rule.params.get("theta_prune", rule.preconditions.get("theta_prune", self.theta_prune)))
            r = float(g.get_reliability(eid))
            if r > theta:
                return DPO_ApplyResult(applied=False, reason=f"Reliability {r:.6f} > theta_prune {theta:.6f}", changes={})
            ok = g.remove_edge(eid)
            if ok:
                return DPO_ApplyResult(applied=True, reason="Edge removed", changes={"removed_edges": [str(eid)]})
            else:
                return DPO_ApplyResult(applied=False, reason="Removal failed", changes={})

        elif kind == "FREEZE":
            # Precondition: edge exists and reliability >= theta
            eid_raw = rule.params.get("edge_id", match.edge_id)
            if eid_raw is None:
                return DPO_ApplyResult(applied=False, reason="Missing edge_id", changes={})
            eid = EdgeId(str(eid_raw))
            if g.get_edge(eid) is None:
                return DPO_ApplyResult(applied=False, reason="Edge not found", changes={})
            theta = float(rule.params.get("theta_freeze", rule.preconditions.get("theta_freeze", self.theta_freeze)))
            r = float(g.get_reliability(eid))
            if r < theta:
                return DPO_ApplyResult(applied=False, reason=f"Reliability {r:.6f} < theta_freeze {theta:.6f}", changes={})
            try:
                g.set_edge_attr(eid, "frozen", True)
                return DPO_ApplyResult(applied=True, reason="Edge frozen", changes={"state_changes": [{"edge_id": str(eid), "attr": "frozen", "value": True}]})
            except Exception:
                return DPO_ApplyResult(applied=False, reason="Freeze failed", changes={})

        else:
            return DPO_ApplyResult(applied=False, reason=f"Unknown rule kind '{kind}'", changes={})


__all__ = ["DPO_LKR", "DPO_Rule", "DPO_Match", "DPO_ApplyResult", "DPOGraphAdapter", "DPOEngine"]