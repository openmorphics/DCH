from __future__ import annotations

from dch_core.interfaces import Event, SeedConfig, make_vertex_id
from dch_pipeline.pipeline import DCHPipeline, PipelineConfig, PlasticityConfig
from dch_pipeline.replay import set_global_seeds, get_environment_fingerprint


def test_set_global_seeds_idempotent():
    seeds = SeedConfig(python=123, numpy=456, torch=789)
    a = set_global_seeds(seeds)
    b = set_global_seeds(seeds)
    assert a == b
    # Expected keys present with correct types
    assert "python" in a and (a["python"] is None or isinstance(a["python"], int))
    assert "numpy" in a and (a["numpy"] is None or isinstance(a["numpy"], int))
    assert "torch" in a and (a["torch"] is None or isinstance(a["torch"], int))


def test_pipeline_metrics_deterministic_single_step():
    # Same seeds applied before each fresh run should yield identical metrics
    seeds = SeedConfig(python=42, numpy=42, torch=42)

    def run_once():
        set_global_seeds(seeds)
        cfg = PipelineConfig(
            plasticity=PlasticityConfig(impl="beta"),
            audit_log_path=None,  # ensure no EAT logging side effects
        )
        connectivity_map = {10: [1, 2]}
        pipeline, _ = DCHPipeline.from_defaults(cfg=cfg, connectivity_map=connectivity_map)

        # Deterministic synthetic events within DHG window [delay_min, delay_max] = [100, 500]
        events = [
            Event(neuron_id=1, t=600),   # presyn (Δ=400)
            Event(neuron_id=2, t=900),   # presyn (Δ=100)
            Event(neuron_id=10, t=1000), # head
        ]
        target_vid = make_vertex_id(10, 1000)

        metrics = pipeline.step(events, target_vertices=[target_vid], sign=+1)
        return metrics

    m1 = run_once()
    m2 = run_once()
    assert m1 == m2

    # Environment fingerprint is side-effect free and consistent in type
    fp = get_environment_fingerprint()
    assert isinstance(fp.python_version, str) and len(fp.python_version) > 0
    assert isinstance(fp.platform, str) and len(fp.platform) > 0