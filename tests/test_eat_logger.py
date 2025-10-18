from __future__ import annotations

import json
from datetime import datetime

from dch_pipeline.eat_logger import EATAuditLogger, verify_file
from dch_core.interfaces import Hyperpath, Event, make_vertex_id
from dch_pipeline.pipeline import DCHPipeline, PipelineConfig


def _iso_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def test_emit_and_verify_chain_ok(tmp_path):
    log_path = tmp_path / "eat.jsonl"
    logger = EATAuditLogger(str(log_path))

    # Emit mixed records
    logger.emit_grow(["e1", "e2"], now_t_iso=_iso_now())

    hp = Hyperpath(
        head=make_vertex_id(10, 1000),
        edges=("e1", "e2"),
        score=0.9,
        length=2,
        label="L1",
    )
    logger.emit_eat(hp, now_t_us=1000)

    logger.emit_update({"e1": 0.5, "e2": 0.75}, now_t_iso=_iso_now())

    # Verify chain
    lines = log_path.read_text(encoding="utf-8").splitlines()
    res = logger.verify()
    assert res["ok"] is True
    assert res["count"] == len(lines)

    # Also check module-level verifier
    res2 = verify_file(str(log_path))
    assert res2["ok"] is True
    assert res2["count"] == len(lines)


def test_tamper_breaks_chain(tmp_path):
    log_path = tmp_path / "eat_tamper.jsonl"
    logger = EATAuditLogger(str(log_path))

    # Write three records
    logger.emit_grow(["a"], now_t_iso=_iso_now())
    logger.emit_update({"a": 0.2}, now_t_iso=_iso_now())
    hp = Hyperpath(head=make_vertex_id(5, 500), edges=("a",), score=0.3, length=1, label=None)
    logger.emit_eat(hp, now_t_us=500)

    # Tamper with the second line's payload (keep hash unchanged)
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 3
    tamper_idx = 1  # zero-based
    rec = json.loads(lines[tamper_idx])
    # Flip a value in payload deterministically
    if rec.get("kind") == "UPDATE" and "payload" in rec and "edges" in rec["payload"] and rec["payload"]["edges"]:
        rec["payload"]["edges"][0]["reliability"] = rec["payload"]["edges"][0]["reliability"] + 0.1
    elif rec.get("kind") == "GROW" and "payload" in rec:
        rec["payload"]["count"] = rec["payload"].get("count", 0) + 1
    else:
        # generic tamper
        rec["ts"] = rec["ts"].replace("Z", "")  # still valid string, but changes hash input

    lines[tamper_idx] = json.dumps(rec, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    res = verify_file(str(log_path))
    assert res["ok"] is False
    assert isinstance(res["bad_index"], int)
    assert res["bad_index"] == tamper_idx


def test_pipeline_emits_when_configured(tmp_path):
    # Configure pipeline with audit logger
    log_path = tmp_path / "pipeline_audit.jsonl"
    cfg = PipelineConfig(audit_log_path=str(log_path))

    connectivity_map = {10: [1, 2]}
    pipeline, _encoder = DCHPipeline.from_defaults(cfg=cfg, connectivity_map=connectivity_map)

    # Events: presyn (1,2) then head (10)
    events = [
        Event(neuron_id=1, t=500),
        Event(neuron_id=2, t=600),
        Event(neuron_id=10, t=1000),
    ]

    # Traverse from the head in the same step; id is known deterministically
    v_head_id = make_vertex_id(10, 1000)
    _metrics = pipeline.step(events, target_vertices=[v_head_id], sign=+1)

    # Verify audit file exists and contains expected kinds
    assert log_path.exists()
    lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) >= 3, "Expected at least one GROW, one EAT, and one UPDATE record"

    kinds = {json.loads(l)["kind"] for l in lines}
    assert {"GROW", "EAT", "UPDATE"}.issubset(kinds), f"Missing kinds, got: {kinds}"

    res = verify_file(str(log_path))
    assert res["ok"] is True
    assert res["count"] == len(lines)