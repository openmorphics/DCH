# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Quick experiment runner CLI (P2-3).

Provides a minimal, reproducible entry-point to execute a deterministic synthetic
streaming run using existing evaluation helpers.

- Determinism:
  Seeds are passed through to the evaluation helper, which applies them to
  Python/NumPy/torch (when available) for reproducible results.

- Artifacts:
  A single metrics JSONL record is written to <artifacts_dir>/metrics.jsonl
  containing config_fingerprint, environment, metrics, and reliability summary.

- Scope:
  Only 'synthetic' mode is implemented in this task (P2-3). A dataset-backed
  mode is future work and will be introduced in P2-4.

Usage:
  python -m scripts.run_quick_experiment --mode synthetic --artifacts-dir artifacts/quick
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Any


# Reuse existing helpers (no heavy deps; stdlib-only at this layer)
from dch_pipeline.evaluation import run_quick_synthetic  # noqa: E402


def run_main(
    mode: str = "synthetic",
    artifacts_dir: str = "artifacts/quick",
    seed_py: int = 123,
    seed_np: int = 123,
    seed_torch: int = 0,
) -> dict:
    """
    Execute a quick experiment and return the result dict.

    This function is intentionally minimal and non-invasive. It constructs a spec
    compatible with dch_pipeline.evaluation.run_quick_synthetic(), triggers the
    run, and returns the helper's result.

    Determinism:
    - Seeds are forwarded to the evaluation helper, which applies them using
      dch_pipeline.replay.set_global_seeds() and captures a stable environment
      fingerprint via dch_pipeline.replay.get_environment_fingerprint().

    Artifacts:
    - A single JSONL line is emitted to <artifacts_dir>/metrics.jsonl with keys:
      {"ts","config_fingerprint","env","metrics","reliability_summary"}

    Future work:
    - Dataset-backed mode will be introduced in P2-4.

    Args:
        mode: Execution mode. Only "synthetic" is supported in P2-3.
        artifacts_dir: Directory to store artifacts (JSONL).
        seed_py: Python random seed.
        seed_np: NumPy random seed.
        seed_torch: PyTorch random seed.

    Returns:
        dict with {"metrics","reliability_summary","artifact"} as produced by
        run_quick_synthetic().

    Raises:
        NotImplementedError: if mode is not "synthetic".
    """
    if mode != "synthetic":
        raise NotImplementedError("mode must be 'synthetic' for P2-3")

    # Build a minimal, deterministic spec understood by run_quick_synthetic()
    spec: Dict[str, Any] = {
        "seeds": {"python": int(seed_py), "numpy": int(seed_np), "torch": int(seed_torch)},
        "pipeline_overrides": {
            "plasticity": {"impl": "beta"},
            "dhg": {"delay_min": 100, "delay_max": 500},
        },
        "connectivity": {"2": [1]},
    }

    return run_quick_synthetic(spec, artifacts_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_quick_experiment",
        description="Run a deterministic synthetic quick experiment and emit a metrics JSONL artifact.",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="synthetic",
        choices=["synthetic"],
        help="Execution mode. Only 'synthetic' is supported in P2-3.",
    )
    p.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/quick",
        help="Directory to write artifacts into (default: artifacts/quick).",
    )
    p.add_argument("--seed-py", type=int, default=123, help="Python random seed (default: 123).")
    p.add_argument("--seed-np", type=int, default=123, help="NumPy random seed (default: 123).")
    p.add_argument("--seed-torch", type=int, default=0, help="PyTorch random seed (default: 0).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_main(
        mode=args.mode,
        artifacts_dir=args.artifacts_dir,
        seed_py=args.seed_py,
        seed_np=args.seed_np,
        seed_torch=args.seed_torch,
    )
    # Single-line JSON on stdout
    print(json.dumps(result, sort_keys=True))


__all__ = ["run_main"]