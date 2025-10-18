# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Confluence (critical-pair) and bounded termination tests for the minimal DPO engine.

Deterministic, offline, quick:
- Uses small in-memory snapshots built by analysis helpers
- Reuses DPO primitives; stdlib-only
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from dch_core.dpo import DPO_Rule, DPO_LKR, DPO_Match
from dch_core.dpo_analysis import (
    analyze_critical_pairs,
    check_termination,
)

# -------------------------
# Utilities (local to tests)
# -------------------------

def _vid(s: str) -> str:
    # VertexId-like helper (string form "neuron@time")
    return s


def _seed_edge(
    eid: str,
    tail: Sequence[str],
    head: str,
    *,
    s: int = 0,
    f: int = 0,
    reliability: float = 0.10,
    frozen: bool = False,
    delta_min: int = 0,
    delta_max: int = 0,
) -> Tuple[Sequence[str], str, Dict[str, Any]]:
    """
    Seed edge tuple for clone_adapter_with:
      (tails, head, attrs)
    """
    return (
        list(tail),
        head,
        {
            "id": eid,
            "counts_success": int(s),
            "counts_miss": int(f),
            "reliability": float(reliability),
            "frozen": bool(frozen),
            "delta_min": int(delta_min),
            "delta_max": int(delta_max),
        },
    )


# -------------------------
# Confluence tests
# -------------------------

def test_confluence_prune_then_freeze_joinable():
    """
    Seed graph: single edge e with posterior mean ~0.75 (s=8, f=2), not frozen.
    Rules:
      - A = FREEZE with theta_freeze=0.7 (applies)
      - B = PRUNE with theta_prune=0.2 (does not apply since 0.75 > 0.2)
    Expect:
      - A∘B and B∘A yield identical final state (joinable), and report contains a reason string.
    """
    tail = [_vid("1@1000")]
    head = _vid("2@1500")
    seed = _seed_edge("e", tail, head, s=8, f=2, reliability=0.10, frozen=False)

    A = DPO_Rule(
        name="freeze_e",
        kind="FREEZE",
        lkr=DPO_LKR(),
        preconditions={"theta_freeze": 0.7},
        params={"edge_id": "e", "theta_freeze": 0.7},
    )
    B = DPO_Rule(
        name="prune_e",
        kind="PRUNE",
        lkr=DPO_LKR(),
        preconditions={"theta_prune": 0.2},
        params={"edge_id": "e", "theta_prune": 0.2},
    )

    matches_by_rule = {
        "freeze_e": [DPO_Match(edge_id="e")],
        "prune_e": [DPO_Match(edge_id="e")],
        "__seed_edges__": [seed],
    }

    rep = analyze_critical_pairs([A, B], matches_by_rule, max_pairs=10)
    assert rep["total_pairs"] >= 2, "Should evaluate ordered pairs (A,B) and (B,A)"
    # Find the A->B specific report row
    rows = [p for p in rep["pairs"] if p["A"] == "freeze_e" and p["B"] == "prune_e"]
    assert rows, "Expected a pair entry for (freeze_e, prune_e)"
    row = rows[0]
    assert row["joinable"] is True, f"Expected joinable, got reason={row['reason']}"
    assert isinstance(row["reason"], str) and len(row["reason"]) > 0


def test_confluence_disjoint_operations_joinable():
    """
    Seed graph: edges e1 and e2 on distinct heads.
    Rules:
      - A = GROW a new edge between disjoint vertices
      - B = PRUNE e1 (low posterior mean to ensure removal)
    Expect:
      - A∘B and B∘A commute (joinable True).
    """
    # Existing edges
    e1 = _seed_edge("e1", [_vid("3@1000")], _vid("10@1500"), s=0, f=10, reliability=0.90, frozen=False)
    e2 = _seed_edge("e2", [_vid("4@1200")], _vid("11@1600"), s=5, f=0, reliability=0.90, frozen=False)

    # A: GROW a brand-new edge (ensure vertices are disjoint)
    new_tail = [_vid("5@1100")]
    new_head = _vid("12@1700")
    A = DPO_Rule(
        name="grow_new",
        kind="GROW",
        lkr=DPO_LKR(),
        preconditions={},
        params={"tails": new_tail, "head": new_head, "attributes": {"note": "disjoint"}},
    )
    # B: PRUNE e1 with a moderate threshold to remove (posterior mean (1+0)/(2+0+10)=1/12≈0.083)
    B = DPO_Rule(
        name="prune_e1",
        kind="PRUNE",
        lkr=DPO_LKR(),
        preconditions={"theta_prune": 0.5},
        params={"edge_id": "e1", "theta_prune": 0.5},
    )

    matches_by_rule = {
        "grow_new": [DPO_Match()],          # empty vertices -> engine uses params tails/head
        "prune_e1": [DPO_Match(edge_id="e1")],
        "__seed_edges__": [e1, e2],
        # No need for extra vertices; analysis utility ensures vertices from params exist
    }

    rep = analyze_critical_pairs([A, B], matches_by_rule, max_pairs=10)
    rows = [p for p in rep["pairs"] if p["A"] == "grow_new" and p["B"] == "prune_e1"]
    assert rows, "Expected a pair entry for (grow_new, prune_e1)"
    assert rows[0]["joinable"] is True, f"Expected joinable, got reason={rows[0]['reason']}"


# -------------------------
# Bounded termination tests
# -------------------------

def test_termination_no_grow_is_terminating():
    # Only PRUNE/FREEZE present
    rules = [
        DPO_Rule(name="P", kind="PRUNE", lkr=DPO_LKR(), params={}, preconditions={}),
        DPO_Rule(name="F", kind="FREEZE", lkr=DPO_LKR(), params={}, preconditions={}),
    ]
    rep = check_termination(rules)
    assert rep["terminate"] is True, f"Expected terminate=True, got {rep}"


def test_termination_with_grow_bounded():
    rules = [
        DPO_Rule(name="G", kind="GROW", lkr=DPO_LKR(), params={}, preconditions={}),
        DPO_Rule(name="P", kind="PRUNE", lkr=DPO_LKR(), params={}, preconditions={}),
    ]
    rep = check_termination(rules, max_grow=2)
    assert rep["terminate"] is True, f"Expected terminate=True with budget, got {rep}"


def test_termination_unknown_when_unbounded_grow():
    rules = [
        DPO_Rule(name="G", kind="GROW", lkr=DPO_LKR(), params={}, preconditions={}),
        DPO_Rule(name="F", kind="FREEZE", lkr=DPO_LKR(), params={}, preconditions={}),
    ]
    rep = check_termination(rules, max_grow=None)
    assert rep["terminate"] == "unknown", f"Expected 'unknown' termination, got {rep['terminate']}"