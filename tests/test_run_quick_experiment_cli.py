# tests/test_run_quick_experiment_cli.py
"""
Smoke tests for the quick experiment runner CLI (P2-3).

Validates:
- run_main(mode="synthetic", ...) emits an artifact JSONL with required fields.
- Deterministic outputs across identical seeds to different artifact dirs.

Notes:
- Uses only stdlib and existing helpers indirectly via scripts/run_quick_experiment.py.
- Dataset-backed mode is future work (P2-4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

# Robust import: prefer package import, fall back to file import if needed
try:
    from scripts.run_quick_experiment import run_main  # type: ignore[attr-defined]
except Exception:
    import importlib.util as _ilu
    import sys

    _ROOT = Path(__file__).resolve().parent.parent
    _MOD_PATH = _ROOT / "scripts" / "run_quick_experiment.py"
    spec = _ilu.spec_from_file_location("run_quick_experiment", str(_MOD_PATH))
    assert spec and spec.loader
    mod = _ilu.module_from_spec(spec)
    sys.modules["run_quick_experiment"] = mod
    spec.loader.exec_module(mod)  # type: ignore[assignment]
    run_main: Callable[..., dict] = getattr(mod, "run_main")


def test_run_main_synthetic_emits_artifact(tmp_path: Path) -> None:
    out_dir = tmp_path / "artifacts"
    res = run_main(
        mode="synthetic",
        artifacts_dir=str(out_dir),
        seed_py=123,
        seed_np=123,
        seed_torch=0,
    )

    # Result surface keys
    assert isinstance(res, dict)
    for k in ("metrics", "reliability_summary", "artifact"):
        assert k in res

    # Artifact exists and is JSONL with required keys
    artifact_path = Path(res["artifact"])
    assert artifact_path.exists() and artifact_path.is_file()
    data = artifact_path.read_text(encoding="utf-8").splitlines()
    assert len(data) >= 1
    rec = json.loads(data[0])
    # Minimal required fields for this task
    for k in ("config_fingerprint", "metrics", "reliability_summary"):
        assert k in rec


def test_run_main_synthetic_is_deterministic(tmp_path: Path) -> None:
    a1 = tmp_path / "a1"
    a2 = tmp_path / "a2"
    r1 = run_main(
        mode="synthetic",
        artifacts_dir=str(a1),
        seed_py=123,
        seed_np=123,
        seed_torch=0,
    )
    r2 = run_main(
        mode="synthetic",
        artifacts_dir=str(a2),
        seed_py=123,
        seed_np=123,
        seed_torch=0,
    )
    assert r1["metrics"] == r2["metrics"]
    assert r1["reliability_summary"] == r2["reliability_summary"]