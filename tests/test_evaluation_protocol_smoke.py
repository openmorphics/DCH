# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Smoke tests for P2 reproducible experiment protocol utilities.

Validates:
- run_quick_synthetic emits deterministic metrics and a JSONL artifact.
- summarize_reliability via run_quick_synthetic returns CIs and means.
"""

from __future__ import annotations

import json
from pathlib import Path

from dch_pipeline.evaluation import run_quick_synthetic, config_fingerprint


def _build_spec() -> dict:
    return {
        "seeds": {"python": 123, "numpy": 123, "torch": 0},
        "pipeline_overrides": {
            "plasticity": {"impl": "beta"},
            "dhg": {"delay_min": 100, "delay_max": 500},
        },
        "connectivity": {"2": [1]},  # head neuron 2 has presyn 1
    }


def test_run_quick_synthetic_emits_artifact_and_summary(tmp_path):
    spec = _build_spec()
    out_dir = tmp_path / "artifacts"
    res = run_quick_synthetic(spec, str(out_dir))

    # Metrics keys present
    metrics = res.get("metrics", {})
    for k in ("n_candidates", "n_admitted", "n_hyperpaths", "n_edges_updated"):
        assert k in metrics

    # Reliability summary structure
    rs = res.get("reliability_summary", {})
    assert "mean_reliability" in rs
    assert "edges" in rs and isinstance(rs["edges"], list) and len(rs["edges"]) >= 1
    for e in rs["edges"]:
        assert set(e.keys()) >= {"id", "mean", "ci"}
        lo, hi = e["ci"]
        m = e["mean"]
        assert 0.0 <= lo < hi <= 1.0
        assert lo <= m <= hi

    # Artifact exists and contents validate
    metrics_path = Path(res["artifact"])
    assert metrics_path.exists() and metrics_path.is_file()
    first = metrics_path.read_text(encoding="utf-8").strip().splitlines()[0]
    rec = json.loads(first)
    for k in ("ts", "config_fingerprint", "env", "metrics", "reliability_summary"):
        assert k in rec
    # Fingerprint reproducibility
    assert rec["config_fingerprint"] == config_fingerprint(spec)


def test_run_quick_synthetic_deterministic_metrics(tmp_path):
    spec = _build_spec()
    out1 = tmp_path / "a1"
    out2 = tmp_path / "a2"
    r1 = run_quick_synthetic(spec, str(out1))
    r2 = run_quick_synthetic(spec, str(out2))
    assert r1["metrics"] == r2["metrics"]
    assert r1["reliability_summary"] == r2["reliability_summary"]