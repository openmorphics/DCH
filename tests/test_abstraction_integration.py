
# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Integration test: wire Streaming FSM promotions to hierarchical abstraction via the pipeline (torch-free).

Construct a tiny two-step chain in the in-memory hypergraph:
    {A, B} --e1--> X --e2--> Y

Run DCHPipeline.step() multiple times with enable_abstraction=True and low FSM thresholds so that
the FSM promotes the observed hyperpath label, and verify that the abstraction engine creates
a higher-order hyperedge (HOE) with head=Y and tail={A,B}. Also verify dedup/idempotence and
that the pipeline runs without abstraction enabled.
"""

from __future__ import annotations

import math

from dch_pipeline.pipeline import DCHPipeline, PipelineConfig, FSMConfig  # [dch_pipeline/pipeline.py:PipelineConfig()]
from dch_core.interfaces import (  # [dch_core/interfaces.py:Event()]
    Event,
    Hyperedge,
    Vertex,
    VertexId,
    EdgeId,
    make_edge_id,
)
from dch_core.abstraction import AbstractionParams  # [dch_core/abstraction.py:AbstractionParams()]


def _build_chain_in_pipeline(
    pipeline: DCHPipeline,
    *,
    r1: float = 0.8,
    r2: float = 0.6,
    t_shift: int = 0,
) -> tuple[Vertex, Vertex, Vertex, Vertex, Hyperedge, Hyperedge]:
    """
    Build and insert a two-step chain inside the pipeline's in-memory hypergraph:

        {A, B} --e1--> X --e2--> Y

    Returns: (vA, vB, vX, vY, e1, e2)
    """
    # Timestamps
    tA = 100 + t_shift
    tB = 120 + t_shift
    tX = 200 + t_shift
    tY = 300 + t_shift

    # Ingest vertices (events)
    vA = pipeline.hypergraph.ingest_event(Event(neuron_id=1, t=tA))
    vB = pipeline.hypergraph.ingest_event(Event(neuron_id=2, t=tB))
    vX = pipeline.hypergraph.ingest_event(Event(neuron_id=3, t=tX))
    vY = pipeline.hypergraph.ingest_event(Event(neuron_id=4, t=tY))

    # e1: {A,B} -> X with window covering observed delays
    dtA = vX.t - vA.t
    dtB = vX.t - vB.t
    delta_min_e1 = min(dtA, dtB)
    delta_max_e1 = max(dtA, dtB)
    tail_e1 = {vA.id, vB.id}
    e1 = Hyperedge(
        id=make_edge_id(head=vX.id, tail=tail_e1, t=vX.t),
        tail=tail_e1,
        head=vX.id,
        delta_min=delta_min_e1,
        delta_max=delta_max_e1,
        refractory_rho=0,
        reliability=r1,
        provenance="e1",
    )

    # e2: {X} -> Y with exact delay
    dtX = vY.t - vX.t
    tail_e2 = {vX.id}
    e2 = Hyperedge(
        id=make_edge_id(head=vY.id, tail=tail_e2, t=vY.t),
        tail=tail_e2,
        head=vY.id,
        delta_min=dtX,
        delta_max=dtX,
        refractory_rho=0,
        reliability=r2,
        provenance="e2",
    )

    # Insert edges
    _ = pipeline.hypergraph.insert_hyperedges([e1, e2])

    return vA, vB, vX, vY, e1, e2


def _find_hoe_AB_to_Y(pipeline: DCHPipeline, vY: Vertex, vA: Vertex, vB: Vertex) -> Hyperedge | None:
    """Find the unique HOE with head Y and tail exactly {A,B}, if it exists."""
    tail_target = {vA.id, vB.id}
    for eid in pipeline.hypergraph.get_incoming_edges(vY.id):
        e = pipeline.hypergraph.get_edge(eid)
        if e is None:
            continue
        if e.head == vY.id and set(e.tail) == tail_target:
            return e
    return None


def _count_hoe_AB_to_Y(pipeline: DCHPipeline, vY: Vertex, vA: Vertex, vB: Vertex) -> int:
    """Count HOEs with head Y and tail {A,B}."""
    cnt = 0
    tail_target = {vA.id, vB.id}
    for eid in pipeline.hypergraph.get_incoming_edges(vY.id):
        e = pipeline.hypergraph.get_edge(eid)
        if e is None:
            continue
        if e.head == vY.id and set(e.tail) == tail_target:
            cnt += 1
    return cnt


def test_abstraction_wiring_promotes_two_step_chain_into_hoe():
    # Configure pipeline with abstraction enabled and very permissive FSM thresholds
    # Keep it torch-free; score approx r1*r2*0.98^2 ~ 0.8*0.6*0.9604 ~ 0.461
    cfg = PipelineConfig(
        enable_abstraction=True,
        fsm=FSMConfig(theta=0.2, lambda_decay=0.0, hold_k=2, min_weight=1e-6, promotion_limit_per_step=10),
        # Back-compat alias for cap
        fsm_promotion_limit_per_step=10,
    )
    pipeline, _enc = DCHPipeline.from_defaults(cfg=cfg)

    # Build the two-step chain in the pipeline
    # Build the two-step chain in the pipeline
    vA, vB, vX, vY, e1, e2 = _build_chain_in_pipeline(pipeline, r1=0.8, r2=0.6, t_shift=0)

    # No HOE exists yet
    assert _find_hoe_AB_to_Y(pipeline, vY, vA, vB) is None

    # Step 1: observe hyperpath via traversal (should not promote yet due to hysteresis hold_k=2)
    metrics1 = pipeline.step(events=[], target_vertices=[vY.id], sign=+1)
    assert isinstance(metrics1, dict)
    assert _find_hoe_AB_to_Y(pipeline, vY, vA, vB) is None

    # Step 2: second observe should cross threshold and promote
    metrics2 = pipeline.step(events=[], target_vertices=[vY.id], sign=+1)
    assert isinstance(metrics2, dict)
    hoe = _find_hoe_AB_to_Y(pipeline, vY, vA, vB)
    assert hoe is not None, "Expected promoted higher-order edge A,B-&gt;Y after FSM promotion"

    # Verify head/tail
    assert hoe.head == vY.id
    assert set(hoe.tail) == {vA.id, vB.id}

    # Aggregated reliability uses AbstractionParams default "min" clamped by floor
    expected_rel = max(AbstractionParams().reliability_floor, min(float(e1.reliability), float(e2.reliability)))
    assert math.isclose(float(hoe.reliability), float(expected_rel), rel_tol=1e-9, abs_tol=1e-12)

    # Idempotence: if promotion occurs again for same label, no duplicate edges are created
    n_before = _count_hoe_AB_to_Y(pipeline, vY, vA, vB)
    # Promotion resets hysteresis; do two more observes to re-trigger
    _ = pipeline.step(events=[], target_vertices=[vY.id], sign=+1)
    _ = pipeline.step(events=[], target_vertices=[vY.id], sign=+1)
    n_after = _count_hoe_AB_to_Y(pipeline, vY, vA, vB)
    assert n_after == n_before == 1, f"Abstraction should dedup A,B-&gt;Y; found {n_after}"



def test_pipeline_sanity_without_abstraction():
    """Pipeline should run without errors when enable_abstraction=False."""
    cfg = PipelineConfig(enable_abstraction=False)  # [PipelineConfig()](dch_pipeline/pipeline.py:1)
    pipeline, _enc = DCHPipeline.from_defaults(cfg=cfg)
    metrics = pipeline.step(
        events=[Event(neuron_id=1, t=10), Event(neuron_id=2, t=20)],
        target_vertices=None,
    )
    assert isinstance(metrics, dict)
    # Basic keys remain present for backward compatibility
    for k in ["n_events_ingested", "n_vertices_new", "n_candidates", "n_admitted"]:
        assert k in metrics