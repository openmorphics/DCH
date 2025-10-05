#!/usr/bin/env python3
# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
DCH experiment runner (scaffold)

Provides a minimal CLI to:
- set seeds and deterministic flags
- construct a default DCH pipeline (in-memory backend)
- generate a synthetic event stream (temporarily, until dataset loaders land)
- run one or more steps with optional credit assignment targets
- write artifacts (metrics.csv, config.json, env.json) under artifacts/<run_id>/

Usage examples (CPU):
- python scripts/run_experiment.py --experiment synthetic --backend norse --device cpu --epochs 3

References:
- Pipeline orchestrator: dch_pipeline.pipeline.DCHPipeline (see [dch_pipeline.pipeline.DCHPipeline.__init__()](dch_pipeline/pipeline.py:97))
- Seeding utilities: [dch_pipeline.seeding.set_global_seeds()](dch_pipeline/seeding.py:27), [dch_pipeline.seeding.enable_torch_determinism()](dch_pipeline/seeding.py:55)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import sys

from dch_core.interfaces import Event, VertexId, make_vertex_id
from dch_pipeline.pipeline import DCHPipeline, PipelineConfig

# Optional seeding imports
if TORCH_AVAILABLE:
    from dch_pipeline.seeding import set_global_seeds, enable_torch_determinism


def _now_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _append_metrics_csv(path: Path, row: Dict[str, Any]) -> None:
    header = list(row.keys())
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _collect_env_fingerprint() -> Dict[str, Any]:
    # Minimal environment fingerprint; extend with more fields as needed
    info: Dict[str, Any] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": os.uname().sysname if hasattr(os, "uname") else "unknown",
        "torch_available": TORCH_AVAILABLE,
    }
    if TORCH_AVAILABLE:
        info.update({
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name_0": (torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else None),
        })
    return info


def _synthetic_events() -> Sequence[Event]:
    """
    Build a small synthetic event sequence (5 events) spanning 3 neurons.
    The temporal structure is designed to admit simple TC-kNN candidates and allow a backward traversal.
    """
    return [
        Event(neuron_id=0, t=1000),  # presyn
        Event(neuron_id=1, t=1400),  # presyn
        Event(neuron_id=2, t=2000),  # postsyn
        Event(neuron_id=2, t=2600),  # postsyn
        Event(neuron_id=1, t=3100),  # presyn (target)
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a DCH experiment (scaffold).")
    p.add_argument("--experiment", type=str, default="synthetic", help="synthetic|dvs_gesture|nmnist (TBD)")
    p.add_argument("--backend", type=str, default="norse", help="norse|bindsnet (unused in scaffold)")
    p.add_argument("--device", type=str, default="cpu", help="cpu|cuda:0")
    p.add_argument("--epochs", type=int, default=1, help="number of epochs (steps) to run")
    p.add_argument("--seed", type=int, default=123, help="global seed")
    p.add_argument("--artifacts_dir", type=str, default="artifacts", help="root for run outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Seeds and determinism (optional torch)
    if TORCH_AVAILABLE:
        set_global_seeds(args.seed)
        enable_torch_determinism(deterministic=True, cudnn_deterministic=True, cudnn_benchmark=False)
        device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    else:
        # Basic seeding without torch
        import random
        random.seed(args.seed)
        device = "cpu"

    # Build default pipeline (in-memory)
    cfg = PipelineConfig()
    pipeline, _encoder = DCHPipeline.from_defaults(cfg=cfg, connectivity_map={1: [0], 2: [0, 1]})

    # Prepare artifacts
    run_id = _now_run_id()
    out_root = Path(args.artifacts_dir) / run_id
    _ensure_dir(out_root)
    metrics_csv = out_root / "metrics.csv"
    _write_json(out_root / "config.json", {"experiment": args.experiment, "backend": args.backend, "device": args.device, "seed": args.seed, "pipeline_config": {
        "dhg": asdict(cfg.dhg),
        "traversal": asdict(cfg.traversal),
        "plasticity": asdict(cfg.plasticity),
    }})
    _write_json(out_root / "env.json", _collect_env_fingerprint())

    # Data (scaffold)
    if args.experiment == "synthetic":
        events = _synthetic_events()
        target: Optional[VertexId] = make_vertex_id(1, 3100)
        targets = [target] if target is not None else None
    else:
        raise NotImplementedError(f"Experiment '{args.experiment}' not implemented yet in scaffold. Use --experiment synthetic.")

    # Run epochs
    for epoch in range(1, args.epochs + 1):
        metrics = pipeline.step(events=events, target_vertices=targets, sign=+1, freeze_plasticity=False)
        # Add epoch to metrics and append to CSV
        row = {"epoch": epoch, **{k: int(v) for k, v in metrics.items()}}
        _append_metrics_csv(metrics_csv, row)

    print(f"[DCH] Run complete. Artifacts at: {out_root}")


if __name__ == "__main__":
    main()