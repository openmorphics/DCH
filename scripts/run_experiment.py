#!/usr/bin/env python3
# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
DCH experiment runner (Hydra-style config loader, import-safe, torch-optional).

Features
- Deterministic defaults with seed control; no network I/O.
- Loads portable YAML configs (Hydra-style layout) from:
  - configs/pipeline.yaml
  - configs/experiments/{nmnist|dvs_gesture}.yaml
  - configs/model/norse_lif.yaml
- Dotlist overrides via CLI (e.g., 'experiment=nmnist snn.enabled=true model=norse_lif').
- Optional SNN path (Norse) when snn.enabled=true; otherwise DCH-only path.
- Small deterministic evaluation loop to produce accuracy/macro F1 (toy), plus DCH metrics.
- Logging to CSV/JSONL on disk; prints one-line JSON summary to stdout.

Actionable error exits (code=2):
- When SNN path is requested but torch/norse are missing.
- When numpy is missing (needed for metric computation).
- When a named model is requested but model config cannot be found.

References
- Pipeline orchestrator: [dch_pipeline.pipeline.DCHPipeline](dch_pipeline/pipeline.py:146)
- Seeds and determinism: [dch_pipeline.seeding.set_global_seeds()](dch_pipeline/seeding.py:27),
  [dch_pipeline.seeding.enable_torch_determinism()](dch_pipeline/seeding.py:61)
- Evaluation utilities (numpy-only): [dch_pipeline.evaluation.evaluate_predictions()](dch_pipeline/evaluation.py:147)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from importlib import util as ilu
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml  # pyyaml (runtime dep)

from dch_core.interfaces import Event, VertexId, make_vertex_id
from dch_pipeline.pipeline import (
    DCHPipeline,
    PipelineConfig,
    DHGConfig,
    TraversalConfig,
    PlasticityConfig,
    FSMConfig,
    SNNConfig,
)
from dch_pipeline.seeding import enable_torch_determinism, set_global_seeds
from dch_pipeline.logging_utils import ExperimentLogger, make_run_dir
from dch_pipeline.evaluation import evaluate_predictions


# -------------------------
# Utilities
# -------------------------


def _base_dir() -> Path:
    # Resolve project root assuming this script resides under ./scripts/
    here = Path(__file__).resolve()
    return here.parent.parent


def _configs_dir() -> Path:
    return _base_dir() / "configs"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping at top-level.")
    return data


def _nested_set(d: MutableMapping[str, Any], keys: Sequence[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]  # type: ignore[assignment]
    cur[keys[-1]] = value


def _parse_dotlist(argv: Sequence[str]) -> Dict[str, Any]:
    """
    Parse key=value pairs (dotlist style) from argv.
    Example: ["experiment=nmnist", "snn.enabled=true", "model=norse_lif"]
    """
    overrides: Dict[str, Any] = {}
    for tok in argv:
        if "=" not in tok:
            continue
        key, val = tok.split("=", 1)
        key = key.strip()
        val = val.strip()
        # Basic type coercion
        if val.lower() in ("true", "false"):
            coerced: Any = (val.lower() == "true")
        else:
            try:
                if "." in val:
                    coerced = float(val)
                    if coerced.is_integer():
                        coerced = int(coerced)
                else:
                    coerced = int(val)
            except Exception:
                coerced = val
        _nested_set(overrides, key.split("."), coerced)
    return overrides


def _deep_update(dst: Dict[str, Any], src: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(dst)
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_update(out[k], v)  # type: ignore[index]
        else:
            out[k] = v
    return out


def _cfg_to_pipeline_config(cfg_map: Mapping[str, Any]) -> PipelineConfig:
    # Map 'dch' (yaml) -> DHGConfig (dataclass)
    dch_map = cfg_map.get("dch", {}) or {}
    traversal_map = cfg_map.get("traversal", {}) or {}
    plasticity_map = cfg_map.get("plasticity", {}) or {}
    fsm_map = cfg_map.get("fsm", {}) or {}
    scaffolding_map = cfg_map.get("scaffolding", {}) or {}
    snn_map = cfg_map.get("snn", {}) or {}

    dhg = DHGConfig(
        k=int(dch_map.get("k", DHGConfig.k)),
        combination_order_max=int(dch_map.get("combination_order_max", DHGConfig.combination_order_max)),
        causal_coincidence_delta=int(dch_map.get("causal_coincidence_delta", DHGConfig.causal_coincidence_delta)),
        budget_per_head=int(dch_map.get("budget_per_head", DHGConfig.budget_per_head)),
        init_reliability=float(dch_map.get("init_reliability", DHGConfig.init_reliability)),
        refractory_rho=int(dch_map.get("refractory_rho", DHGConfig.refractory_rho)),
        delay_min=int(dch_map.get("delay_min", DHGConfig.delay_min)),
        delay_max=int(dch_map.get("delay_max", DHGConfig.delay_max)),
    )
    trav = TraversalConfig(
        horizon=int(traversal_map.get("horizon", TraversalConfig.horizon)),
        beam_size=int(traversal_map.get("beam_size", TraversalConfig.beam_size)),
        length_penalty_base=float(traversal_map.get("length_penalty_base", TraversalConfig.length_penalty_base)),
    )
    plast = PlasticityConfig(
        ema_alpha=float(plasticity_map.get("ema_alpha", PlasticityConfig.ema_alpha)),
        reliability_min=float(plasticity_map.get("reliability_min", PlasticityConfig.reliability_min)),
        reliability_max=float(plasticity_map.get("reliability_max", PlasticityConfig.reliability_max)),
        decay_lambda=float(plasticity_map.get("decay_lambda", PlasticityConfig.decay_lambda)),
        prune_threshold=float(plasticity_map.get("prune_threshold", PlasticityConfig.prune_threshold)),
    )
    fsm = FSMConfig(
        theta=float(fsm_map.get("theta", FSMConfig.theta)),
        lambda_decay=float(fsm_map.get("lambda_decay", FSMConfig.lambda_decay)),
        hold_k=int(fsm_map.get("hold_k", FSMConfig.hold_k)),
        min_weight=float(fsm_map.get("min_weight", FSMConfig.min_weight)),
        promotion_limit_per_step=int(fsm_map.get("promotion_limit_per_step", FSMConfig.promotion_limit_per_step)),
    )
    snn = SNNConfig(
        enabled=bool(snn_map.get("enabled", SNNConfig.enabled)),
        model=str(snn_map.get("model", SNNConfig.model)),
        unroll=bool(snn_map.get("unroll", SNNConfig.unroll)),
        device=str(snn_map.get("device", SNNConfig.device)),
        model_params=dict(snn_map.get("model_params", {})),
    )

    enable_abstraction = bool(scaffolding_map.get("enabled", False))
    abstraction_params = {}  # reserved

    return PipelineConfig(
        dhg=dhg,
        traversal=trav,
        plasticity=plast,
        fsm=fsm,
        enable_abstraction=enable_abstraction,
        abstraction_params=abstraction_params,
        fsm_promotion_limit_per_step=fsm.promotion_limit_per_step,
        snn=snn,
    )


def _synthetic_events() -> Sequence[Event]:
    # Small deterministic sequence spanning 3 neurons with causal structure
    return [
        Event(neuron_id=0, t=1000),  # presyn
        Event(neuron_id=1, t=1400),  # presyn
        Event(neuron_id=2, t=2000),  # postsyn
        Event(neuron_id=2, t=2600),  # postsyn
        Event(neuron_id=1, t=3100),  # presyn (target)
    ]


def _resolve_experiment_name(ovr: Mapping[str, Any]) -> str:
    exp = ovr.get("experiment", "synthetic")
    if isinstance(exp, Mapping):
        # allow experiment.name=...
        name = exp.get("name", "synthetic")
        return str(name)
    return str(exp)


def _load_model_cfg(model_name: Optional[str]) -> Dict[str, Any]:
    if not model_name:
        return {}
    if model_name == "norse_lif":
        path = _configs_dir() / "model" / "norse_lif.yaml"
        if not path.exists():
            _exit2(f"Model '{model_name}' requested but config file not found at {path}")
        return _load_yaml(path)
    _exit2(f"Unknown model '{model_name}'. Supported: 'norse_lif'.")


def _exit2(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    sys.stderr.write(msg.rstrip() + "\n")
    sys.exit(2)


# -------------------------
# Main
# -------------------------


def main() -> None:
    # Parse overrides from sys.argv (skip program name)
    overrides = _parse_dotlist(sys.argv[1:])

    # Load base pipeline YAML
    base_yaml_path = _configs_dir() / "pipeline.yaml"
    if not base_yaml_path.exists():
        _exit2(f"Missing base config at {base_yaml_path}.")
    base_map = _load_yaml(base_yaml_path)

    # Merge overrides at top-level
    merged = _deep_update(base_map, overrides)

    # Resolve experiment selection
    exp_name = _resolve_experiment_name(merged)
    exp_yaml: Dict[str, Any] = {}
    if exp_name in ("nmnist", "dvs_gesture"):
        exp_path = _configs_dir() / "experiments" / f"{exp_name}.yaml"
        if exp_path.exists():
            exp_yaml = _load_yaml(exp_path)
            merged = _deep_update(merged, exp_yaml)
        else:
            # No hard failure; continue and fall back to synthetic data later.
            pass

    # Resolve model config if requested by 'model=norse_lif'
    model_name = merged.get("model")
    model_map: Dict[str, Any] = {}
    if isinstance(model_name, str) and model_name:
        model_map = _load_model_cfg(model_name)
        # Merge into 'snn.model_params' (forwarded to norse factory)
        if "snn" not in merged:
            merged["snn"] = {}
        # Attach model config tree under model_params for the factory; keep explicit flags too
        merged["snn"]["model_params"] = _deep_update(
            dict(merged["snn"].get("model_params", {})),
            model_map,
        )

    # Determine seed and output locations
    seeds = list(merged.get("experiment", {}).get("seeds", [123])) if isinstance(merged.get("experiment"), dict) else [123]
    seed = int(seeds[0]) if seeds else 123
    artifacts_dir = merged.get("experiment", {}).get("artifacts_dir", "artifacts") if isinstance(merged.get("experiment"), dict) else "artifacts"

    # Dependency gates
    # - numpy for metrics
    if ilu.find_spec("numpy") is None:
        _exit2(
            "Optional dependency 'numpy' is required for metrics.\n"
            "- Try: pip install numpy\n"
            "- Or:  conda install -c conda-forge numpy"
        )
    # - torch/norse if SNN path is enabled
    snn_enabled = bool(merged.get("snn", {}).get("enabled", False))
    if snn_enabled:
        missing = []
        if ilu.find_spec("torch") is None:
            missing.append("torch")
        if ilu.find_spec("norse") is None:
            missing.append("norse")
        if missing:
            _exit2(
                "SNN is enabled but required optional dependencies are missing: "
                + ", ".join(missing)
                + "\n- Try: pip install 'torch>=2.2' 'norse>=0.0.9'\n"
                  "- Or:  conda install -c conda-forge pytorch norse\n"
                  "Alternatively run with 'snn.enabled=false'."
            )

    # Seeds and deterministic knobs
    set_global_seeds(seed)
    enable_torch_determinism(deterministic=True, cudnn_deterministic=True, cudnn_benchmark=False)

    # Build PipelineConfig from merged YAML
    pipe_cfg = _cfg_to_pipeline_config(merged)

    # Construct pipeline and encoder
    # Provide a small connectivity for synthetic runs; real datasets should provide this via experiment configs in future.
    pipeline, encoder = DCHPipeline.from_defaults(
        cfg=pipe_cfg,
        connectivity_map={1: [0], 2: [0, 1]},
    )

    # Data selection (no network I/O; lazy; fallback to synthetic when loaders unavailable)
    events: Sequence[Event]
    target: Optional[VertexId]
    dataset_used = exp_name
    fallback_reason: Optional[str] = None

    if exp_name in ("nmnist", "dvs_gesture"):
        # Try tonic presence; otherwise fall back
        if ilu.find_spec("tonic") is None:
            events = _synthetic_events()
            target = make_vertex_id(1, 3100)
            dataset_used = "synthetic_fallback"
            fallback_reason = "tonic not installed; run scripts/download_datasets.py after installing tonic to enable datasets."
        else:
            # Dataset loader stubs (no IO): Fall back by design for this subtask
            events = _synthetic_events()
            target = make_vertex_id(1, 3100)
            dataset_used = "synthetic_fallback"
            fallback_reason = "dataset loader stubbed in this subtask. No network I/O performed."
    else:
        events = _synthetic_events()
        target = make_vertex_id(1, 3100)

    # Execute a tiny deterministic evaluation loop (single step)
    # DCH metrics
    metrics = pipeline.step(events=events, target_vertices=[target] if target is not None else None, sign=+1, freeze_plasticity=False)

    # Prepare toy classification metrics deterministically:
    # Use C = max(2, N) classes from encoder-derived N when available; predict exactly the true labels for determinism.
    try:
        # Determine device if torch is present
        device_obj = None
        if ilu.find_spec("torch") is not None:
            import torch as _torch  # lazy
            device_obj = _torch.device(pipe_cfg.snn.device if (_torch.cuda.is_available() or "cpu" in str(pipe_cfg.snn.device)) else "cpu")
        spikes, meta = encoder.encode(events, (min(e.t for e in events), max(e.t for e in events)), device_obj)
        N = int(meta.get("N", 0))
    except Exception:
        N = 0
    C = max(2, N if N > 0 else 3)
    # Build y_true deterministically from event neuron ids modulo C, then y_pred == y_true
    y_true = [int(e.neuron_id) % C for e in events]
    y_pred = list(y_true)

    eval_res = evaluate_predictions(y_true, y_pred=y_pred, num_classes=C)
    acc = float(eval_res.metrics["accuracy"])
    macro_f1 = float(eval_res.metrics["macro_f1"])

    # Prepare run directory and loggers
    out_root = make_run_dir(Path(artifacts_dir), prefix=f"{exp_name}")
    logger = ExperimentLogger(
        csv_path=out_root / "metrics.csv",
        jsonl_path=out_root / "metrics.jsonl",
        tb_log_dir=None,
    )
    # Log CSV (single row) and JSONL
    row = {"epoch": 1, **{k: int(v) if isinstance(v, bool | int) else (float(v) if isinstance(v, float) else v) for k, v in metrics.items()}}
    logger.log_csv(row)
    logger.log_jsonl({"metrics": row})
    # Persist merged config for provenance
    with (out_root / "config.merged.json").open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, sort_keys=True)

    # One-line JSON summary to stdout
    summary = {
        "status": "ok",
        "experiment": exp_name,
        "dataset_used": dataset_used,
        "fallback_reason": fallback_reason,
        "seed": seed,
        "snn_enabled": snn_enabled,
        "metrics": {
            "accuracy": acc,
            "macro_f1": macro_f1,
            # Include a couple of DCH counters for quick smoke checks
            "n_events_ingested": int(metrics.get("n_events_ingested", 0)),
            "n_vertices_new": int(metrics.get("n_vertices_new", 0)),
            "n_candidates": int(metrics.get("n_candidates", 0)),
            "n_admitted": int(metrics.get("n_admitted", 0)),
        },
        "artifacts_dir": str(out_root),
        "notes": "Deterministic toy evaluation; dataset loaders are stubbed in this subtask.",
    }
    sys.stdout.write(json.dumps(summary, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()