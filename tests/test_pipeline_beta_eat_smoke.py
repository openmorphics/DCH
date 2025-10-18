from __future__ import annotations

import json

from dch_core.interfaces import Event, make_vertex_id
from dch_pipeline.pipeline import DCHPipeline, PipelineConfig, PlasticityConfig
from dch_pipeline.eat_logger import verify_file


def test_pipeline_smoke_beta_eat_log_integrity(tmp_path):
    # Minimal postsynaptic connectivity: neuron 10 receives from neurons 1 and 2
    connectivity_map = {10: [1, 2]}
    log_path = tmp_path / "eat.jsonl"

    # Configure Beta plasticity and enable EAT audit logging
    cfg = PipelineConfig(
        plasticity=PlasticityConfig(impl="beta"),
        audit_log_path=str(log_path),
    )
    pipeline, _ = DCHPipeline.from_defaults(cfg=cfg, connectivity_map=connectivity_map)

    # Synthesize deterministic events within DHG window [delay_min, delay_max] = [100, 500]
    events = [
        Event(neuron_id=1, t=600),   # presyn (Δ=400)
        Event(neuron_id=2, t=900),   # presyn (Δ=100)
        Event(neuron_id=10, t=1000), # head
    ]
    target_vid = make_vertex_id(10, 1000)

    metrics = pipeline.step(events, target_vertices=[target_vid], sign=+1)

    # Activity across pipeline stages
    assert metrics["n_candidates"] >= 1
    assert metrics["n_admitted"] >= 1
    assert metrics["n_hyperpaths"] >= 1
    assert metrics["n_edges_updated"] >= 1

    # Audit log existence and contents
    assert log_path.exists() and log_path.stat().st_size > 0
    kinds = set()
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            k = rec.get("kind")
            if isinstance(k, str):
                kinds.add(k)

    assert {"GROW", "EAT", "UPDATE"}.issubset(kinds), f"kinds present: {kinds}"
    v = verify_file(str(log_path))
    assert v.get("ok") is True
    assert int(v.get("count", 0)) >= 3


def test_pipeline_smoke_beta_respects_prune_threshold_minimal():
    # High prune threshold should remove low-reliability edges created this step
    connectivity_map = {10: [1]}
    cfg = PipelineConfig(
        plasticity=PlasticityConfig(impl="beta", prune_threshold=0.9),
    )
    pipeline, _ = DCHPipeline.from_defaults(cfg=cfg, connectivity_map=connectivity_map)

    events = [
        Event(neuron_id=1, t=600),   # presyn (Δ=400)
        Event(neuron_id=10, t=1000), # head
    ]
    target_vid = make_vertex_id(10, 1000)

    metrics = pipeline.step(events, target_vertices=[target_vid], sign=+1)

    assert metrics["n_admitted"] >= 1
    assert metrics["n_pruned"] >= 1