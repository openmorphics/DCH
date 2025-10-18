from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Sequence

# Reuse existing evaluation helpers (stdlib-only at this layer)
from dch_pipeline.evaluation import (
    run_quick_synthetic,
    effect_size_time_series,
    aggregate_runs,
)


def _read_first_jsonl_record(path: str) -> Mapping[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline()
    return json.loads(line)


def run_main(
    *,
    artifacts_root: str,
    replicates: int = 2,
    base_seed_py: int = 123,
    base_seed_np: int = 123,
    base_seed_torch: int = 0,
    plasticity: List[str] = None,
    abstraction: List[bool] = None,
    mode: str = "synthetic",
    align_by: str = "name",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    P2-6 Ablation suite runner (opt-in, deterministic). Currently supports synthetic mode.

    Factors:
      - Plasticity: EMA vs Beta via pipeline_overrides.plasticity.impl in the quick synthetic runner
      - Abstraction: on/off flag. For synthetic mode here, abstraction is a stub label; it is included
        in condition names but does not alter behavior (future datasets path may toggle actual pipeline knobs).

    Determinism:
      - Seeds are computed deterministically per replicate: base_seed + r for python/numpy/torch.
      - Each condition/replicate writes artifacts into artifacts_root/<label>/rep<r>/metrics.jsonl

    Artifacts layout:
      artifacts_root/
        p=<impl>|abs=<0/1>/
          rep0/metrics.jsonl
          rep1/metrics.jsonl
          ...

    Returns:
      {
        "alpha": float,
        "conditions": {label: {"n": int, "artifacts": list[str]}},
        "aggregate": aggregate_runs_result,
        "comparisons": [
          {"pair": [label_a, label_b], "effect_sizes": {...}},
          ...
        ],
      }

    Notes:
      - Dataset ablations are future P2 extensions; this task only implements synthetic mode.
      - No subprocess usage; this is a callable entry-point for tests.
    """
    if plasticity is None:
        plasticity = ["beta", "ema"]
    # Normalize plasticity names
    plasticity = [str(p).lower() for p in plasticity]

    if abstraction is None:
        abstraction = [False]
    # Coerce abstraction entries to booleans
    abstraction_bools: List[bool] = [bool(int(a)) if isinstance(a, (int, str)) else bool(a) for a in abstraction]

    if mode != "synthetic":
        raise ValueError(f"Unsupported mode '{mode}' in this task. Only 'synthetic' is supported.")

    os.makedirs(artifacts_root, exist_ok=True)

    # Accumulators
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    conditions: Dict[str, Dict[str, Any]] = {}

    # 1) Run synthetic quick experiments for each factor combination and replicate
    for p in plasticity:
        for a in abstraction_bools:
            label = f"p={p}|abs={int(a)}"
            for r in range(int(replicates)):
                seed_py = int(base_seed_py) + r
                seed_np = int(base_seed_np) + r
                seed_torch = int(base_seed_torch) + r

                spec: Dict[str, Any] = {
                    "seeds": {"python": seed_py, "numpy": seed_np, "torch": seed_torch},
                    "pipeline_overrides": {
                        "plasticity": {"impl": p},
                        # Abstraction is stubbed for synthetic mode; include flag for labeling only
                        "enable_abstraction": bool(a),
                        "dhg": {"delay_min": 100, "delay_max": 500},
                    },
                    "connectivity": {"2": [1]},
                }

                out_dir = os.path.join(artifacts_root, label, f"rep{r}")
                os.makedirs(out_dir, exist_ok=True)
                res = run_quick_synthetic(spec, out_dir)

                metrics_path = str(res.get("artifact"))
                rec = _read_first_jsonl_record(metrics_path)

                # Inject a stable label for indexing and human-readable reporting
                if "__name" not in rec:
                    rec["__name"] = label

                groups.setdefault(label, []).append(rec)
                if label not in conditions:
                    conditions[label] = {"n": 0, "artifacts": []}
                conditions[label]["n"] = int(conditions[label]["n"]) + 1
                conditions[label]["artifacts"].append(metrics_path)

    # 2) Aggregate reliability summaries across groups
    agg = aggregate_runs(groups, alpha=float(alpha))

    # 3) Compute pairwise effect sizes for Beta vs EMA within the same abstraction value
    comparisons: List[Dict[str, Any]] = []
    abs_values = sorted({int(l.split("|abs=")[1]) for l in groups.keys() if "|abs=" in l})
    for a_int in abs_values:
        label_beta = f"p=beta|abs={a_int}"
        label_ema = f"p=ema|abs={a_int}"
        if label_beta in groups and label_ema in groups:
            recs_beta = groups[label_beta]
            recs_ema = groups[label_ema]
            eff = effect_size_time_series(recs_beta, recs_ema, align_by=str(align_by), alpha=float(alpha))
            # Fallback to index alignment if name alignment produced no pairs (common when names differ)
            if (isinstance(eff, dict) and isinstance(eff.get("pairs"), list) and len(eff["pairs"]) == 0) and str(align_by) == "name":
                eff_idx = effect_size_time_series(recs_beta, recs_ema, align_by="index", alpha=float(alpha))
                eff = eff_idx
            comparisons.append({"pair": [label_beta, label_ema], "effect_sizes": eff})

    return {
        "alpha": float(alpha),
        "conditions": conditions,
        "aggregate": agg,
        "comparisons": comparisons,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2-6 Ablation suite: Beta vs EMA and abstraction toggles (synthetic mode).")
    parser.add_argument("--artifacts-root", type=str, required=True, help="Root directory to write artifacts.")
    parser.add_argument("--replicates", type=int, default=2, help="Number of replicates per condition (default: 2).")
    parser.add_argument("--base-seed-py", type=int, default=123, help="Base Python seed (default: 123).")
    parser.add_argument("--base-seed-np", type=int, default=123, help="Base NumPy seed (default: 123).")
    parser.add_argument("--base-seed-torch", type=int, default=0, help="Base torch seed (default: 0).")
    parser.add_argument(
        "--plasticity",
        type=str,
        nargs="*",
        default=["beta", "ema"],
        help='Plasticity implementations to compare (default: "beta ema").',
    )
    parser.add_argument(
        "--abstraction",
        type=int,
        nargs="*",
        default=[0],
        help="Abstraction flags as ints 0/1 (default: 0).",
    )
    parser.add_argument("--mode", type=str, choices=["synthetic"], default="synthetic", help="Experiment mode (only 'synthetic' supported).")
    parser.add_argument("--align-by", type=str, choices=["index", "name"], default="name", help="Alignment for effect sizes (default: name).")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha for CI-to-sigma approximation (default: 0.05).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_main(
        artifacts_root=str(args.artifacts_root),
        replicates=int(args.replicates),
        base_seed_py=int(args.base_seed_py),
        base_seed_np=int(args.base_seed_np),
        base_seed_torch=int(args.base_seed_torch),
        plasticity=[str(p) for p in args.plasticity],
        abstraction=[int(a) for a in args.abstraction],
        mode=str(args.mode),
        align_by=str(args.align_by),
        alpha=float(args.alpha),
    )
    # Single-line JSON for easy piping/parsing
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())