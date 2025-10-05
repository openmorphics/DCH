from __future__ import annotations

from dch_core.interfaces import Event
from dch_pipeline.pipeline import DCHPipeline, PipelineConfig


def test_pipeline_smoke_no_torch():
    """Test that pipeline works without torch dependency."""
    # Create pipeline with default in-memory backends
    pipeline, encoder = DCHPipeline.from_defaults()
    
    # Generate synthetic events
    events = [
        Event(neuron_id=1, t=100),
        Event(neuron_id=2, t=200), 
        Event(neuron_id=10, t=1000),  # head event
    ]
    
    # Run one pipeline step
    metrics = pipeline.step(events)
    
    # Verify basic metrics structure
    assert "n_events_ingested" in metrics
    assert "n_vertices_new" in metrics
    assert "n_candidates" in metrics
    assert "n_admitted" in metrics
    assert metrics["n_events_ingested"] == 3
    assert metrics["n_vertices_new"] == 3
    assert metrics["n_candidates"] >= 0
    assert metrics["n_admitted"] >= 0


def test_pipeline_with_target_vertices():
    """Test pipeline with supervised targets for credit assignment."""
    # Setup connectivity: neuron 10 receives from neurons 1,2
    connectivity_map = {10: [1, 2]}
    pipeline, encoder = DCHPipeline.from_defaults(
        connectivity_map=connectivity_map
    )
    
    # Generate events with temporal structure
    events = [
        Event(neuron_id=1, t=500),   # presyn
        Event(neuron_id=2, t=600),   # presyn  
        Event(neuron_id=10, t=1000), # postsyn (head)
    ]
    
    # Run step with target for credit assignment
    v_head = pipeline.hypergraph.ingest_event(events[-1])
    metrics = pipeline.step(
        events[:-1],  # only presyn events in this step
        target_vertices=[v_head.id],
        sign=+1  # positive evidence
    )
    
    # Should have traversal and plasticity metrics
    assert "n_hyperpaths" in metrics
    assert "n_edges_updated" in metrics  
    assert "n_pruned" in metrics
    assert metrics["n_hyperpaths"] >= 0
    assert metrics["n_edges_updated"] >= 0
    assert metrics["n_pruned"] >= 0