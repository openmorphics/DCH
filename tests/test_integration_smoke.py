# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Integration smoke test for the DCH pipeline.

Covers:
- In-memory hypergraph backend
- TC-kNN candidate generation (basic run)
- Backward traversal call path
- Plasticity update/prune no-ops on tiny data
"""

from __future__ import annotations

import pytest
pytest.importorskip("torch")

from dch_core.interfaces import Event, VertexId, make_vertex_id
from dch_pipeline.pipeline import DCHPipeline, PipelineConfig


def test_pipeline_smoke_cpu_defaults():
    # Build pipeline with a tiny connectivity: 0 -> 1, 0,1 -> 2
    pipe, _enc = DCHPipeline.from_defaults(
        cfg=PipelineConfig(),
        connectivity_map={1: [0], 2: [0, 1]},
    )

    # Construct a tiny event sequence (timestamps in arbitrary microseconds)
    events = [
        Event(neuron_id=0, t=1000),
        Event(neuron_id=1, t=1400),
        Event(neuron_id=2, t=2000),
        Event(neuron_id=2, t=2600),
        Event(neuron_id=1, t=3100),
    ]

    # Target: last event's vertex id
    target_vid: VertexId = make_vertex_id(1, 3100)

    # Single-step run: ingest + DHG + traversal + plasticity
    metrics = pipe.step(events=events, target_vertices=[target_vid], sign=+1, freeze_plasticity=False)

    # Basic checks on metrics presence and types
    assert isinstance(metrics, dict)
    for key in ["n_events_ingested", "n_vertices_new", "n_candidates", "n_admitted", "n_hyperpaths", "n_edges_updated", "n_pruned"]:
        assert key in metrics, f"missing metric: {key}"
        assert isinstance(metrics[key], int), f"metric {key} must be int-like"

    # Non-negativity
    assert metrics["n_events_ingested"] >= 0
    assert metrics["n_vertices_new"] >= 0
    assert metrics["n_candidates"] >= 0
    assert metrics["n_admitted"] >= 0
    assert metrics["n_hyperpaths"] >= 0
    assert metrics["n_edges_updated"] >= 0
    assert metrics["n_pruned"] >= 0