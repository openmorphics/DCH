# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Statistical report CLI (P2-5).

Purpose
- Load one or two sets of metrics artifacts (JSONL produced by quick/dataset runners)
- Compute reliability CI time series for set A
- Optionally compute effect sizes (Cohen's d, Hedges' g) between sets A and B
- Optionally emit a single-line JSON report to a specified path

Determinism
- This script only aggregates existing artifacts and computes closed-form/approx values.
- No randomness introduced here; determinism is inherited from artifacts.

Usage
  python -m scripts.run_stats_report \\
    --a artifacts/quick/metrics.jsonl \\
    --a artifacts/dataset/metrics.jsonl \\
    --b artifacts/quick_baseline/metrics.jsonl \\
    --align-by name \\
    --alpha 0.05 \\
    --output reports/stats_report.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from dch_pipeline.evaluation import (  # type: ignore
    reliability_ci_time_series,
    effect_size_time_series,
)


def _coerce_to_records(paths: Sequence[str]) -> List[Mapping[str, Any]]:
    """
    Load JSONL artifact records from a list of paths.

    Path handling:
    - If a path is a directory, look for '<dir>/metrics.jsonl'
    - If a path is a file, read it directly
    - Only the first JSONL line is read (as produced by quick/dataset helpers)
    - Injects a '__name' field when not present using a stable basename

    Returns:
      List of dict-like records suitable for evaluation utilities.
    """
    out: List[Mapping[str, Any]] = []
    for p in paths:
        path = str(p)
        if os.path.isdir(path):
            path = os.path.join(path, "metrics.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact path not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
        if not first:
            raise ValueError(f"Artifact file is empty: {path}")
        rec = json.loads(first)
        # Provide a stable fallback name for alignment if missing
        if "__name" not in rec or not isinstance(rec.get("__name"), str) or not rec.get("__name"):
            # Prefer directory name if path is .../<dir>/metrics.jsonl
            base_dir = os.path.basename(os.path.dirname(path))
            base_file = os.path.splitext(os.path.basename(path))[0]
            label = base_dir if base_dir else base_file
            rec["__name"] = label
        out.append(rec)
    return out


def run_main(
    a: Sequence[str],
    b: Optional[Sequence[str]] = None,
    *,
    align_by: str = "index",  # "index" | "name"
    alpha: float = 0.05,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute statistical summaries from artifact sets.

    Args:
      a: Sequence of file/dir paths for set A artifacts.
      b: Optional sequence of file/dir paths for set B artifacts (for effect sizes).
      align_by: Alignment strategy for effect sizes ('index' or 'name').
      alpha: Significance level for CI-to-sigma approximation (default 0.05).
      output_path: Optional path to write a single-line JSON report.

    Returns:
      Dict with at least:
        - 'reliability_ts': output of reliability_ci_time_series(records_a)
        - 'effect_sizes': output of effect_size_time_series(records_a, records_b) when b is provided
        - 'artifact': output_path when provided and written
    """
    if not a:
        raise ValueError("At least one --a path must be provided")

    records_a = _coerce_to_records(a)
    rel_ts = reliability_ci_time_series(records_a, alpha=alpha)
    out: Dict[str, Any] = {"reliability_ts": rel_ts}

    if b:
        records_b = _coerce_to_records(b)
        eff = effect_size_time_series(records_a, records_b, align_by=align_by, alpha=alpha)
        out["effect_sizes"] = eff

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(out, sort_keys=True) + "\n")
        out["artifact"] = output_path

    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_stats_report",
        description="Compute reliability time series and optional effect sizes from DCH artifacts.",
    )
    p.add_argument(
        "--a",
        dest="a",
        action="append",
        default=[],
        help="Path to artifact file or directory (metrics.jsonl). May be provided multiple times.",
    )
    p.add_argument(
        "--b",
        dest="b",
        action="append",
        default=[],
        help="Optional: Path(s) for comparison set (same format as --a). May be provided multiple times.",
    )
    p.add_argument(
        "--align-by",
        dest="align_by",
        choices=["index", "name"],
        default="index",
        help="Alignment for effect sizes: 'index' pairs by order; 'name' pairs by derived names.",
    )
    p.add_argument(
        "--alpha",
        dest="alpha",
        type=float,
        default=0.05,
        help="Alpha for CI-to-sigma approximation (default: 0.05).",
    )
    p.add_argument(
        "--output",
        dest="output",
        type=str,
        default=None,
        help="Optional path to write single-line JSON report.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_main(
        a=args.a,
        b=(args.b or None),
        align_by=args.align_by,
        alpha=args.alpha,
        output_path=args.output,
    )
    print(json.dumps(result, sort_keys=True))


__all__ = ["run_main"]