# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Baselines experiment runner CLI (opt-in; P2-7).

Provides a deterministic, offline-safe entry point to execute a minimal
experiment for Norse SG and BindsNET STDP, writing a JSONL artifact
compatible with existing analysis utilities.

- Seeds applied via dch_pipeline.replay.set_global_seeds().
- Fingerprints via dch_pipeline.evaluation.config_fingerprint().
- Best-effort YAML load; does not fail if PyYAML is missing.
- If baseline module or interface is unavailable, falls back to a dry-run
  synthetic metrics dict.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from typing import Any, Dict, Optional

from dch_core.interfaces import SeedConfig
from dch_pipeline.replay import set_global_seeds
from dch_pipeline.evaluation import config_fingerprint


SUPPORTED_BASELINES = {"norse_sg", "bindsnet_stdp"}


def _resolve_default_config_path(baseline: str, provided: Optional[str]) -> str:
    if provided:
        return str(provided)
    # Prefer baseline-specific config if available, else micro
    baselines_dir = os.path.join("configs", "baselines")
    if baseline == "norse_sg":
        cand = os.path.join(baselines_dir, "norse_sg.yaml")
    else:
        cand = os.path.join(baselines_dir, "bindsnet_stdp.yaml")
    if os.path.exists(cand):
        return cand
    return os.path.join("configs", "micro.yaml")


def _best_effort_yaml_load(path: str) -> dict:
    try:
        yaml = importlib.import_module("yaml")
    except Exception:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = getattr(yaml, "safe_load")(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _try_run_baseline(baseline: str, cfg_map: dict, limit: int) -> Dict[str, Any] | None:
    """
    Attempt to import and run a minimal entry-point from the baseline module.
    Returns metrics dict on success, or None to signal fallback.
    """
    try:
        if baseline == "norse_sg":
            mod = importlib.import_module("baselines.norse_sg")
            # Prefer a generic 'run' if present
            if hasattr(mod, "run"):
                return dict(mod.run(cfg_map, limit=limit))  # type: ignore[attr-defined]
            # Else attempt tiny train_and_eval with max_steps mapping
            if hasattr(mod, "BaselineConfig") and hasattr(mod, "train_and_eval"):
                cfg = getattr(mod, "BaselineConfig")()
                if "seed" in cfg_map:
                    setattr(cfg, "seed", int(cfg_map["seed"]))
                setattr(cfg, "max_steps", int(limit))
                res = getattr(mod, "train_and_eval")(cfg)
                if isinstance(res, dict):
                    return dict(res)
        elif baseline == "bindsnet_stdp":
            mod = importlib.import_module("baselines.bindsnet_stdp")
            if hasattr(mod, "run"):
                return dict(mod.run(cfg_map, limit=limit))  # type: ignore[attr-defined]
            if hasattr(mod, "BaselineConfig") and hasattr(mod, "run_unsupervised"):
                cfg = getattr(mod, "BaselineConfig")()
                if "seed" in cfg_map:
                    setattr(cfg, "seed", int(cfg_map["seed"]))
                setattr(cfg, "max_steps", int(limit))
                res = getattr(mod, "run_unsupervised")(cfg)
                if isinstance(res, dict):
                    return dict(res)
    except Exception:
        # Any failure (missing deps, interface mismatch, runtime error) -> fallback
        return None
    return None


def run_main(
    baseline: str = "norse_sg",
    config_path: Optional[str] = None,
    artifacts_dir: str = "artifacts/baselines",
    seed_py: int = 123,
    seed_np: int = 123,
    seed_torch: int = 0,
    limit: int = 10,
) -> dict:
    """
    Run a minimal baseline experiment and emit a JSONL artifact.

    Returns:
        {"metrics": dict, "reliability_summary": dict, "artifact": str}
    """
    b = str(baseline).lower()
    if b not in SUPPORTED_BASELINES:
        raise ValueError(f"baseline must be one of {sorted(SUPPORTED_BASELINES)}")

    cfg_path = _resolve_default_config_path(b, config_path)
    # Apply seeds deterministically (best-effort if numpy/torch absent)
    seeds = SeedConfig(python=int(seed_py), numpy=int(seed_np), torch=int(seed_torch), extra={})
    _ = set_global_seeds(seeds)

    # Load YAML best-effort (not required for operation)
    cfg_map = _best_effort_yaml_load(cfg_path)

    metrics = _try_run_baseline(b, cfg_map, int(limit))
    if not isinstance(metrics, dict):
        metrics = {"steps": int(limit), "baseline": b, "status": "dry_run"}

    record = {
        "baseline": b,
        "config_path": cfg_path,
        "config_fingerprint": config_fingerprint({"baseline": b, "config_path": cfg_path}),
        "metrics": metrics,
        "reliability_summary": {"mean_reliability": 0.0, "edges": []},
    }

    os.makedirs(str(artifacts_dir), exist_ok=True)
    out_path = os.path.join(str(artifacts_dir), "metrics.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

    return {"metrics": record["metrics"], "reliability_summary": record["reliability_summary"], "artifact": out_path}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_baselines_experiment",
        description="Run a deterministic minimal baseline (Norse SG or BindsNET STDP) and emit a JSONL artifact.",
    )
    p.add_argument("--baseline", type=str, choices=sorted(SUPPORTED_BASELINES), default="norse_sg")
    p.add_argument("--config-path", type=str, default=None)
    p.add_argument("--artifacts-dir", type=str, default="artifacts/baselines")
    p.add_argument("--seed-py", type=int, default=123)
    p.add_argument("--seed-np", type=int, default=123)
    p.add_argument("--seed-torch", type=int, default=0)
    p.add_argument("--limit", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    res = run_main(
        baseline=args.baseline,
        config_path=args.config_path,
        artifacts_dir=args.artifacts_dir,
        seed_py=args.seed_py,
        seed_np=args.seed_np,
        seed_torch=args.seed_torch,
        limit=args.limit,
    )
    print(json.dumps(res, sort_keys=True))