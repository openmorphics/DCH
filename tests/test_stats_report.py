# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
P2-5 Statistical validation harness tests.

Covers:
- reliability_ci_time_series(): builds CI series from artifact records
- effect_size_time_series(): computes Cohen's d and Hedges' g with index | name alignment
- scripts.run_stats_report.run_main(): end-to-end aggregation without subprocess

Notes:
- Uses synthetic artifacts emitted by run_quick_synthetic() to avoid external I/O.
- Align-by "name" path leverages config_fingerprint present in artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

import pytest

from dch_pipeline.evaluation import (  # type: ignore
    run_quick_synthetic,
    reliability_ci_time_series,
    effect_size_time_series,
    config_fingerprint,
)


def _default_spec() -> dict:
    return {
        "seeds": {"python": 123, "numpy": 123, "torch": 0},
        "pipeline_overrides": {
            "plasticity": {"impl": "beta"},
            "dhg": {"delay_min": 100, "delay_max": 500},
        },
        "connectivity": {"2": [1]},  # head neuron 2 has presyn 1
    }


def _emit_artifact(tmp_dir: Path, sub: str, spec: dict) -> Mapping[str, Any]:
    out_dir = tmp_dir / sub
    res = run_quick_synthetic(spec, str(out_dir))
    metrics_path = Path(res["artifact"])
    assert metrics_path.exists() and metrics_path.is_file()
    first = metrics_path.read_text(encoding="utf-8").strip().splitlines()[0]
    rec = json.loads(first)
    return rec


def test_reliability_ci_time_series_from_artifact(tmp_path: Path) -> None:
    spec = _default_spec()
    rec = _emit_artifact(tmp_path, "a1", spec)

    series = reliability_ci_time_series([rec], alpha=0.05)
    assert "alpha" in series and series["alpha"] == pytest.approx(0.05)
    assert "names" in series and isinstance(series["names"], list) and len(series["names"]) == 1
    assert "mean" in series and isinstance(series["mean"], list) and len(series["mean"]) == 1
    assert "ci" in series and isinstance(series["ci"], list) and len(series["ci"]) == 1

    # Validate CI contains mean and is bounded
    m = float(series["mean"][0])
    lo, hi = series["ci"][0]
    assert 0.0 <= lo <= m <= hi <= 1.0

    # Label should default to config_fingerprint when present
    assert series["names"][0] == rec["config_fingerprint"]


def test_effect_size_time_series_align_by_index_and_name(tmp_path: Path) -> None:
    spec = _default_spec()
    rec_a = _emit_artifact(tmp_path, "a", spec)
    rec_b = _emit_artifact(tmp_path, "b", spec)  # identical config -> expect zero effect

    # Align by index
    eff_idx = effect_size_time_series([rec_a], [rec_b], align_by="index", alpha=0.05)
    assert "cohen_d" in eff_idx and len(eff_idx["cohen_d"]) == 1
    assert "hedges_g" in eff_idx and len(eff_idx["hedges_g"]) == 1
    # Identical summaries -> zero effect sizes (guarded by sigma handling)
    assert eff_idx["cohen_d"][0] == pytest.approx(0.0)
    assert eff_idx["hedges_g"][0] == pytest.approx(0.0)

    # Align by name (via config_fingerprint)
    eff_nm = effect_size_time_series([rec_a], [rec_b], align_by="name", alpha=0.05)
    assert "pairs" in eff_nm and len(eff_nm["pairs"]) == 1
    nm_a, nm_b = eff_nm["pairs"][0]
    assert nm_a == rec_a["config_fingerprint"]
    assert nm_b == rec_b["config_fingerprint"]
    assert eff_nm["cohen_d"][0] == pytest.approx(0.0)
    assert eff_nm["hedges_g"][0] == pytest.approx(0.0)


def test_run_stats_report_main_end_to_end(tmp_path: Path) -> None:
    # Robust import: prefer package import, fall back to file import if needed
    try:
        from scripts.run_stats_report import run_main  # type: ignore[attr-defined]
    except Exception:
        import importlib.util as _ilu
        import sys

        _ROOT = Path(__file__).resolve().parent.parent
        _MOD_PATH = _ROOT / "scripts" / "run_stats_report.py"
        spec = _ilu.spec_from_file_location("run_stats_report", str(_MOD_PATH))
        assert spec and spec.loader
        mod = _ilu.module_from_spec(spec)
        sys.modules["run_stats_report"] = mod
        spec.loader.exec_module(mod)  # type: ignore[assignment]
        run_main: Callable[..., dict] = getattr(mod, "run_main")

    spec = _default_spec()
    # Two identical artifacts so effect sizes are defined and near-zero
    rec_a = _emit_artifact(tmp_path, "setA", spec)
    rec_b = _emit_artifact(tmp_path, "setB", spec)

    a_paths = [str((tmp_path / "setA" / "metrics.jsonl"))]
    b_paths = [str((tmp_path / "setB" / "metrics.jsonl"))]
    report_path = tmp_path / "report.jsonl"

    out = run_main(a=a_paths, b=b_paths, align_by="name", alpha=0.05, output_path=str(report_path))
    assert "reliability_ts" in out
    assert "effect_sizes" in out
    assert "artifact" in out and out["artifact"] == str(report_path)

    # Validate the written report
    assert report_path.exists() and report_path.is_file()
    line = report_path.read_text(encoding="utf-8").strip().splitlines()[0]
    rec = json.loads(line)
    assert "reliability_ts" in rec
    assert "effect_sizes" in rec