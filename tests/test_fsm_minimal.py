# Minimal FSM test adapted to current DCH interfaces and FSM engine
from __future__ import annotations

from dch_core.interfaces import Hyperpath, VertexId, EdgeId
from dch_core.fsm import StreamingFSMEngine


def test_streaming_fsm_promotion():
    # Build a simple hyperpath with a fixed label and non-trivial score
    hp = Hyperpath(
        head=VertexId("H"),
        edges=(EdgeId("e1"),),
        score=3.0,
        length=1,
        label="rule1",
    )

    # FSM with threshold 5.0 and hysteresis hold_k=2:
    # After two observes of weight 3.0, count=6.0 and promotion should trigger
    fsm = StreamingFSMEngine(theta=5.0, lambda_decay=0.0, hold_k=2)

    promos_1 = fsm.observe([hp], now_t=1000)
    assert promos_1 == [], "First observation should not yet promote due to hysteresis"

    promos_2 = fsm.observe([hp], now_t=1001)
    assert "rule1" in promos_2, "Second observation should promote the frequent hyperpath label"