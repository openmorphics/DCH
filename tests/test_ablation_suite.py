from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scripts.run_ablation_suite import run_main  # type: ignore
from dch_pipeline.evaluation import run_quick_synthetic, aggregate_runs  # type: ignore


def _emit_artifact(tmp_dir: Path, sub: str, spec: dict) -> Mapping[str, Any]:
    out_dir = tmp_dir / sub
    res = run_quick_synthetic(spec, str(out_dir))
    metrics_path = Path(res["artifact"])
    assert metrics_path.exists() and metrics_path.is_file()
    first = metrics_path.read_text(encoding="utf-8").strip().splitlines()[0]
    rec = json.loads(first)
    return rec


def _default_spec(plasticity_impl: str = "beta", seed_py: int = 123, seed_np: int = 123, seed_torch: int = 0) -> dict:
    return {
        "seeds": {"python": int(seed_py), "numpy": int(seed_np), "torch": int(seed_torch)},
        "pipeline_overrides": {
            "plasticity": {"impl": str(plasticity_impl)},
            "dhg": {"delay_min": 100, "delay_max": 500},
        },
        "connectivity": {"2": [1]},
    }


def test_ablation_suite_synthetic_minimal(tmp_path: Path) -> None:
    out = run_main(
        artifacts_root=str(tmp_path / "abl"),
        replicates=1,
        base_seed_py=123,
        base_seed_np=123,
        base_seed_torch=0,
        plasticity=["beta", "ema"],
        abstraction=[0],
        mode="synthetic",
        align_by="name",
        alpha=0.05,
    )
    # Structure
    for k in ("conditions", "aggregate", "comparisons"):
        assert k in out

    conds = out["conditions"]
    assert "p=beta|abs=0" in conds
    assert "p=ema|abs=0" in conds
    assert int(conds["p=beta|abs=0"]["n"]) == 1
    assert int(conds["p=ema|abs=0"]["n"]) == 1

    # Comparisons must include the Beta vs EMA pair for abs=0
    comps = out.get("comparisons", [])
    found = False
    for c in comps:
        if c.get("pair") == ["p=beta|abs=0", "p=ema|abs=0"]:
            eff = c.get("effect_sizes", {})
            assert "cohen_d" in eff
            assert "hedges_g" in eff
            found = True
            break
    assert found


def test_aggregate_runs_api(tmp_path: Path) -> None:
    # Create two synthetic records in different directories but same seeds/spec
    spec = _default_spec(plasticity_impl="beta", seed_py=123, seed_np=123, seed_torch=0)
    rec1 = _emit_artifact(tmp_path, "a1", spec)
    rec2 = _emit_artifact(tmp_path, "a2", spec)

    groups = {"grpA": [rec1, rec2]}
    agg = aggregate_runs(groups, alpha=0.05)

    assert "alpha" in agg and agg["alpha"] == 0.05
    assert "groups" in agg and "grpA" in agg["groups"]
    g = agg["groups"]["grpA"]
    assert "reliability_ts" in g and "group_mean" in g and "n" in g
    assert int(g["n"]) == 2

    series = g["reliability_ts"]
    assert "names" in series and isinstance(series["names"], list) and len(series["names"]) == 2
    assert "mean" in series and isinstance(series["mean"], list) and len(series["mean"]) == 2
    assert "ci" in series and isinstance(series["ci"], list) and len(series["ci"]) == 2


def test_ablation_suite_determinism(tmp_path: Path) -> None:
    args = dict(
        artifacts_root=str(tmp_path / "abl1"),
        replicates=2,
        base_seed_py=123,
        base_seed_np=123,
        base_seed_torch=0,
        plasticity=["beta", "ema"],
        abstraction=[0],
        mode="synthetic",
        align_by="name",
        alpha=0.05,
    )
    out1 = run_main(**args)

    args2 = dict(args)
    args2["artifacts_root"] = str(tmp_path / "abl2")
    out2 = run_main(**args2)

    # Aggregates should be identical across independent roots when args are identical
    assert out1["aggregate"] == out2["aggregate"]
    # The number of comparisons should also match
    assert len(out1.get("comparisons", [])) == len(out2.get("comparisons", []))