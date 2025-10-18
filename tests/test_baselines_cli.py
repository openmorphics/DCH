# Tests for baselines CLI runner (P2-7)
from __future__ import annotations

import json
from pathlib import Path

from scripts.run_baselines_experiment import run_main


def test_run_baselines_norse_smoke(tmp_path):
    out_dir = tmp_path / "norse"
    res = run_main(
        baseline="norse_sg",
        config_path="configs/micro.yaml",
        artifacts_dir=str(out_dir),
        seed_py=123,
        seed_np=123,
        seed_torch=0,
        limit=5,
    )
    assert {"metrics", "reliability_summary", "artifact"} <= set(res.keys())
    artifact_path = Path(res["artifact"])
    assert artifact_path.exists()
    first = artifact_path.read_text(encoding="utf-8").splitlines()[0]
    rec = json.loads(first)
    for k in ("config_fingerprint", "metrics", "reliability_summary"):
        assert k in rec
    assert "mean_reliability" in rec["reliability_summary"]


def test_run_baselines_bindsnet_smoke(tmp_path):
    out_dir = tmp_path / "bindsnet"
    res = run_main(
        baseline="bindsnet_stdp",
        config_path="configs/micro.yaml",
        artifacts_dir=str(out_dir),
        seed_py=123,
        seed_np=123,
        seed_torch=0,
        limit=5,
    )
    assert {"metrics", "reliability_summary", "artifact"} <= set(res.keys())
    artifact_path = Path(res["artifact"])
    assert artifact_path.exists()
    first = artifact_path.read_text(encoding="utf-8").splitlines()[0]
    rec = json.loads(first)
    for k in ("config_fingerprint", "metrics", "reliability_summary"):
        assert k in rec
    assert "mean_reliability" in rec["reliability_summary"]


def test_run_baselines_determinism(tmp_path):
    out1 = tmp_path / "a1"
    out2 = tmp_path / "a2"
    kwargs = dict(
        baseline="norse_sg",
        config_path="configs/micro.yaml",
        seed_py=777,
        seed_np=777,
        seed_torch=0,
        limit=7,
    )
    r1 = run_main(artifacts_dir=str(out1), **kwargs)
    r2 = run_main(artifacts_dir=str(out2), **kwargs)
    assert r1["metrics"] == r2["metrics"]
    assert r1["reliability_summary"] == r2["reliability_summary"]