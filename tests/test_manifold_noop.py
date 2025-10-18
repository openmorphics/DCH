# Copyright (c) 2025 DCH Maintainers
# License: MIT

from __future__ import annotations

from typing import Dict, List

from dch_pipeline import DCHPipeline, PipelineConfig, ManifoldConfig
from dch_core.interfaces import Event, make_vertex_id
from dch_pipeline.evaluation import summarize_reliability
from dch_core.manifold import NoOpManifold


def _make_midpoint_events(cfg: PipelineConfig, head_neuron: int, tail_neuron: int) -> tuple[List[Event], str]:
    """
    Build a deterministic 2-event sequence:
    - one tail/presyn spike, then one head spike
    - tail timestamp at the midpoint of [delay_min, delay_max] relative to head
    """
    t_head = 10_000
    dmin = int(cfg.dhg.delay_min)
    dmax = int(cfg.dhg.delay_max)
    mid_delay = int((dmin + dmax) // 2)
    if mid_delay < dmin:
        mid_delay = dmin
    if mid_delay > dmax:
        mid_delay = dmax
    t_tail = int(t_head - mid_delay)
    events = [
        Event(neuron_id=tail_neuron, t=t_tail, meta=None),
        Event(neuron_id=head_neuron, t=t_head, meta=None),
    ]
    target_vid = make_vertex_id(head_neuron, t_head)
    return events, target_vid


def test_pipeline_constructs_with_noop_manifold_flag() -> None:
    # Small deterministic connectivity
    connectivity_map: Dict[int, list[int]] = {2: [1]}
    cfg = PipelineConfig(
        manifold=ManifoldConfig(enable=True, impl="noop", log_calls=False)
    )
    pipeline, _enc = DCHPipeline.from_defaults(cfg=cfg, connectivity_map=connectivity_map)
    assert getattr(pipeline, "manifold", None) is not None, "pipeline.manifold should be set when enabled"
    assert pipeline.manifold.name() == "noop", "Expected NoOpManifold to be wired when impl='noop'"


def test_noop_does_not_change_metrics() -> None:
    connectivity_map: Dict[int, list[int]] = {2: [1]}

    # Disabled (default)
    cfg_disabled = PipelineConfig()
    pipe_disabled, _ = DCHPipeline.from_defaults(cfg=cfg_disabled, connectivity_map=connectivity_map)

    # Enabled (noop)
    cfg_enabled = PipelineConfig(manifold=ManifoldConfig(enable=True, impl="noop"))
    pipe_enabled, _ = DCHPipeline.from_defaults(cfg=cfg_enabled, connectivity_map=connectivity_map)

    # Deterministic event pair at midpoint delay
    events_d, target_vid_d = _make_midpoint_events(cfg_disabled, head_neuron=2, tail_neuron=1)
    events_e, target_vid_e = _make_midpoint_events(cfg_enabled, head_neuron=2, tail_neuron=1)

    # One supervised step with positive reinforcement
    metrics_disabled = dict(pipe_disabled.step(events_d, target_vertices=[target_vid_d], sign=+1))
    metrics_enabled = dict(pipe_enabled.step(events_e, target_vertices=[target_vid_e], sign=+1))

    # Reliability summaries (Beta prior defaults in summarize_reliability)
    summary_disabled = summarize_reliability(pipe_disabled.hypergraph, alpha0=1.0, beta0=1.0, ci_level=0.95, max_edges=10)
    summary_enabled = summarize_reliability(pipe_enabled.hypergraph, alpha0=1.0, beta0=1.0, ci_level=0.95, max_edges=10)

    # Exact equality guarantees full backward-compatibility when feature is disabled vs noop
    assert metrics_disabled == metrics_enabled, f"Metrics changed under noop manifold: {metrics_disabled} vs {metrics_enabled}"
    assert summary_disabled == summary_enabled, f"Reliability summaries differ under noop manifold: {summary_disabled} vs {summary_enabled}"


def test_noop_backend_interface_contract() -> None:
    backend = NoOpManifold()
    causes = [{"id": "1@1000", "t": 1000}, {"id": "2@1100", "t": 1100}]  # event-like dicts
    effect = {"id": "3@1200", "t": 1200}
    feasible = backend.check_feasible(causes=causes, effect=effect, context={"note": "test"})
    assert feasible is True

    exp = backend.explain(causes=causes, effect=effect, context=None)
    assert isinstance(exp, dict)
    assert exp.get("type") == "noop"
    assert exp.get("feasible") is True