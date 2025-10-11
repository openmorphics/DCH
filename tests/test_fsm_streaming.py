# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Streaming FSM engine tests (torch-free).

Covers:
- Threshold + hysteresis: no promotion before hold_k observations; promotion when threshold is met.
- Queue semantics: pop_promotions() returns promoted items once and is empty thereafter.

Runtime: < 100ms.
"""

from __future__ import annotations

from dch_core.interfaces import Hyperpath, VertexId, EdgeId
from dch_core.fsm import StreamingFSMEngine


def test_streaming_fsm_threshold_and_hold_behavior():
    # Hyperpath with fixed label and weight (score)
    hp = Hyperpath(
        head=VertexId("H"),
        edges=(EdgeId("e1"),),
        score=3.0,
        length=1,
        label="rule1",
    )

    # Threshold 5.0, hysteresis hold_k=2
    fsm = StreamingFSMEngine(theta=5.0, lambda_decay=0.0, hold_k=2)

    # First observation: count=3.0, hold=1 -> no promotion
    promos_1 = fsm.observe([hp], now_t=1000)
    assert promos_1 == [], "No promotion expected before threshold or hold_k satisfied"

    # Second observation: count accumulates to 6.0, hold=2 -> promotion
    promos_2 = fsm.observe([hp], now_t=1001)
    assert "rule1" in promos_2, "Promotion should trigger once threshold and hold_k are satisfied"


def test_pop_promotions_queue_semantics():
    # Use small decay; with two quick observations we still exceed threshold
    fsm = StreamingFSMEngine(theta=4.9, lambda_decay=0.1, hold_k=2)

    hp = Hyperpath(
        head=VertexId("H"),
        edges=(EdgeId("eX"),),
        score=3.0,
        length=1,
        label="ruleX",
    )

    # First observe: no promotion yet
    _ = fsm.observe([hp], now_t=2000)
    # Second observe: promotion occurs (decay applied but still breaches threshold)
    promos = fsm.observe([hp], now_t=2001)
    assert promos == ["ruleX"], "Expected immediate promotion on second observe with decay"

    # pop_promotions should return the same label once, then be empty
    popped_1 = fsm.pop_promotions()
    assert popped_1 == ["ruleX"]
    popped_2 = fsm.pop_promotions()
    assert popped_2 == []