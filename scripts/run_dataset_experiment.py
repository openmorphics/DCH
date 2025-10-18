# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Dataset streaming experiment CLI (opt-in; P2-4).

Overview
- Provides a minimal, deterministic entry point to run a tiny streaming experiment
  labeled for a dataset ("nmnist" or "dvs_gesture") without downloading real data.
- Reuses evaluation helper [run_quick_dataset()](dch_pipeline/evaluation.py:1) to orchestrate a
  short run, persist a metrics JSONL artifact, and return a compact result dict.

Determinism
- Seeds are forwarded to set_global_seeds() within the helper for reproducible results.
- Reliability summary uses fixed-seed MC sampling for stable credible intervals.

Configs and defaults
- If --config-path is omitted, defaults are used based on dataset:
  * nmnist      - configs/experiments/nmnist.yaml
  * dvs_gesture - configs/experiments/dvs_gesture.yaml
- Passing configs/micro.yaml triggers an internal synthetic fallback path to avoid heavy I/O.

Artifacts
- A single JSONL record is written to <artifacts_dir>/metrics.jsonl with keys:
  {"dataset","config_path","config_fingerprint","metrics","reliability_summary"}

Opt-in behavior
- This CLI is non-invasive and does not change default project flows. It is intended
  for quick, deterministic experiments in CI or local development.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

from dch_pipeline.evaluation import run_quick_dataset


def run_main(
    dataset: str = "nmnist",
    config_path: Optional[str] = None,
    artifacts_dir: str = "artifacts/dataset",
    seed_py: int = 123,
    seed_np: int = 123,
    seed_torch: int = 0,
    limit: int = 50,
) -> dict:
    """
    Execute a quick, deterministic dataset-labeled streaming experiment.

    Builds a spec for [run_quick_dataset()](dch_pipeline/evaluation.py:1), invokes it, and returns its result.

    Args:
        dataset: One of {"nmnist","dvs_gesture"}.
        config_path: Optional path to a YAML config. If None, defaults selected per dataset.
        artifacts_dir: Directory to store the JSONL artifact (metrics.jsonl).
        seed_py: Python random seed.
        seed_np: NumPy random seed.
        seed_torch: PyTorch random seed.
        limit: Small cap on number of streaming steps to keep runtime short.

    Returns:
        Dict with {"metrics","reliability_summary","artifact"} as produced by helper.
    """
    spec: Dict[str, Any] = {
        "dataset": str(dataset),
        "config_path": config_path,
        "seeds": {"python": int(seed_py), "numpy": int(seed_np), "torch": int(seed_torch)},
        "pipeline_overrides": {"plasticity": {"impl": "beta"}},
        "limit": int(limit),
    }
    return run_quick_dataset(spec, artifacts_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_dataset_experiment",
        description="Run a deterministic, tiny streaming experiment labeled for a dataset and emit a metrics JSONL artifact.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        choices=["nmnist", "dvs_gesture"],
        default="nmnist",
        help="Dataset label for the experiment (default: nmnist).",
    )
    p.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional path to dataset YAML. If omitted, defaults selected per dataset.",
    )
    p.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/dataset",
        help="Directory to write artifacts into (default: artifacts/dataset).",
    )
    p.add_argument("--seed-py", type=int, default=123, help="Python random seed (default: 123).")
    p.add_argument("--seed-np", type=int, default=123, help="NumPy random seed (default: 123).")
    p.add_argument("--seed-torch", type=int, default=0, help="PyTorch random seed (default: 0).")
    p.add_argument("--limit", type=int, default=50, help="Max steps to emulate (default: 50).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_main(
        dataset=args.dataset,
        config_path=args.config_path,
        artifacts_dir=args.artifacts_dir,
        seed_py=args.seed_py,
        seed_np=args.seed_np,
        seed_torch=args.seed_torch,
        limit=args.limit,
    )
    print(json.dumps(result, sort_keys=True))


__all__ = ["run_main"]