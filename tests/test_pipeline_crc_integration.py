# Copyright (c) 2025 DCH Maintainers
# License: MIT

from __future__ import annotations

import json

from dch_pipeline.pipeline import DCHPipeline, PipelineConfig, FSMConfig
from dch_core.interfaces import Event, Hyperedge, make_edge_id


def _build_two_step_chain(pipeline: DCHPipeline, *, r1: float = 0.8, r2: float = 0.6, t_shift: int = 0):
    """
    Build a tiny chain inside the pipeline hypergraph:

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
        counts_success=10,
        counts_miss=4,
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
        counts_success=7,
        counts_miss=6,
    )

    _ = pipeline.hypergraph.insert_hyperedges([e1, e2])
    return vA, vB, vX, vY, e1, e2


def test_pipeline_crc_logging_on_promotion(tmp_path):
    """
    Wire CRC logging into the pipeline and verify that promotions produce CRC JSONL lines.
    """
    crc_path = tmp_path / "crc.jsonl"
    # Enable abstraction with permissive FSM thresholds to promote quickly
    cfg = PipelineConfig(
        enable_abstraction=True,
        fsm=FSMConfig(theta=0.2, lambda_decay=0.0, hold_k=2, min_weight=1e-6, promotion_limit_per_step=10),
        # Back-compat alias for cap
        fsm_promotion_limit_per_step=10,
        # Opt-in CRC logging
        crc_log_path=str(crc_path),
    )
    pipeline, _ = DCHPipeline.from_defaults(cfg=cfg)

    # Build chain in hypergraph
    vA, vB, vX, vY, e1, e2 = _build_two_step_chain(pipeline, r1=0.8, r2=0.6, t_shift=0)

    # Step 1: traverse from Y (should not promote yet due to hold_k=2)
    _ = pipeline.step(events=[], target_vertices=[vY.id], sign=+1)

    # Step 2: second observation should promote; CRC should be written
    _ = pipeline.step(events=[], target_vertices=[vY.id], sign=+1)

    # Assert CRC log exists and has at least one line
    assert crc_path.exists() and crc_path.stat().st_size > 0
    lines = [ln for ln in crc_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 1

    # Validate presence of required keys in at least one record
    required = {"label", "rule_id", "type", "reliability_mean", "reliability_ci", "text", "provenance_edges"}
    found_valid = False
    for ln in lines:
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        if all(k in rec for k in required):
            # Minimal semantic checks
            assert isinstance(rec["text"], str) and len(rec["text"]) > 0
            ci = rec["reliability_ci"]
            assert isinstance(ci, list) and len(ci) == 2
            assert 0.0 <= float(ci[0]) <= 1.0 <= 1.0
            found_valid = True
            break

    assert found_valid, f"No CRC JSONL line contained required keys among {len(lines)} records"