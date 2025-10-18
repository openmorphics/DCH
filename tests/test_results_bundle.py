import json
import os
from pathlib import Path

from dch_pipeline.evaluation import run_quick_synthetic  # type: ignore
from scripts.make_results_bundle import run_main  # type: ignore
from dch_pipeline.determinism import compare_artifacts  # type: ignore


def _spec_with_seeds():
    # Stable seeds for deterministic synthetic runs
    return {"seeds": {"python": 123, "numpy": 456, "torch": 789}}


def test_bundle_from_quick_synthetic_equal(tmp_path):
    # Produce two deterministic artifacts
    spec = _spec_with_seeds()
    out1 = tmp_path / "a1"
    out2 = tmp_path / "a2"
    r1 = run_quick_synthetic(spec, str(out1))
    r2 = run_quick_synthetic(spec, str(out2))

    # Build a bundle and verify equality
    bundle_dir = tmp_path / "bundle"
    res = run_main([r1["artifact"], r2["artifact"]], bundle_dir=str(bundle_dir), tol=0.0)

    # Basic shape
    assert isinstance(res, dict)
    for k in ("bundle_dir", "concat", "manifest", "comparisons"):
        assert k in res

    # Files exist
    concat_path = Path(res["concat"])
    manifest_path = Path(res["manifest"])
    assert concat_path.exists()
    assert manifest_path.exists()

    # Manifest comparisons: all equal
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "comparisons" in manifest
    assert isinstance(manifest["comparisons"], list)
    assert len(manifest["comparisons"]) == 1
    assert all(c.get("equal") is True for c in manifest["comparisons"])

    # Concatenated JSONL has two lines
    with open(concat_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(lines) == 2


def test_compare_artifacts_tolerance():
    delta = 1e-4

    rec_a = {
        "config_fingerprint": "abc123",
        "metrics": {"m1": 1.0, "flag": True, "label": "ok"},
        "reliability_summary": {
            "mean_reliability": 0.5,
            "edges": [{"id": "e1", "mean": 0.2, "ci": [0.1, 0.3]}],
        },
    }
    rec_b = {
        "config_fingerprint": "abc123",  # same
        "metrics": {"m1": 1.0 + delta, "flag": True, "label": "ok"},  # small numeric delta
        "reliability_summary": {
            "mean_reliability": 0.5 + delta,  # small delta
            "edges": [{"id": "e1", "mean": 0.2 + delta, "ci": [0.1, 0.3]}],  # same CI
        },
    }

    # Tolerance smaller than delta -> not equal
    cmp_small = compare_artifacts(rec_a, rec_b, tol=delta / 10)
    assert cmp_small["equal"] is False
    assert isinstance(cmp_small["diff"], dict)

    # Tolerance larger than delta -> equal
    cmp_big = compare_artifacts(rec_a, rec_b, tol=delta * 10)
    assert cmp_big["equal"] is True


def test_bundle_accepts_dir_inputs(tmp_path):
    spec = _spec_with_seeds()
    d1 = tmp_path / "dir1"
    d2 = tmp_path / "dir2"
    r1 = run_quick_synthetic(spec, str(d1))
    r2 = run_quick_synthetic(spec, str(d2))

    # Pass directories (should resolve to <dir>/metrics.jsonl)
    res = run_main([str(d1), str(d2)], bundle_dir=str(tmp_path / "bundle2"), tol=0.0)

    # Files exist and JSONL has two lines
    concat_path = Path(res["concat"])
    manifest_path = Path(res["manifest"])
    assert concat_path.exists()
    assert manifest_path.exists()
    with open(concat_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(lines) == 2