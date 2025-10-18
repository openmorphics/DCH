"""Make results bundle and verify determinism.

Usage (CLI):
  python -m scripts.make_results_bundle \\
    --artifact path/to/artifacts1/metrics.jsonl \\
    --artifact path/to/artifacts2/metrics.jsonl \\
    --bundle-dir artifacts/bundle \\
    --tol 0.0 \\
    --label "my_bundle"

Concepts:
- Determinism verification:
  Compares multiple artifacts (JSONL, one JSON record per file) for equality.
  Numeric comparisons can use an absolute tolerance (--tol). Strings and booleans
  require exact equality. Extra keys are ignored (comparison uses common keys).
- Tolerance semantics:
  Two floats a and b are considered equal if abs(a - b) <= tol.
- Artifact expectations:
  Each artifact file should contain exactly one JSON object on the first line with:
    {
      "config_fingerprint": str,
      "metrics": dict,
      "reliability_summary": {
        "mean_reliability": float,
        "edges": [{"id": str?, "mean": float, "ci": [lo, hi]}]
      },
      ...
    }
  Additional fields (e.g., "env", "ts") are allowed and may be included in the bundle manifest.

Outputs:
- Concatenated JSONL of artifacts (preserving the input order).
- Manifest JSON summarizing the bundle and comparisons.
- Both are written under the bundle directory.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, List, Mapping, Sequence, Dict, Optional

from dch_pipeline.determinism import load_first_jsonl_record, compare_artifacts, make_manifest


def _resolve_path(p: str) -> str:
    """Resolve artifact file path. If 'p' is a directory, append 'metrics.jsonl'."""
    if os.path.isdir(p):
        p = os.path.join(p, "metrics.jsonl")
    return p


def run_main(
    artifacts: list[str],
    bundle_dir: str = "artifacts/bundle",
    *,
    tol: float = 0.0,
    label: str | None = None,
    concat_name: str = "artifacts.jsonl",
    manifest_name: str = "manifest.json",
) -> dict:
    """
    Build a portable results bundle from one or more artifact paths.

    Steps:
      - Normalize inputs: if a path is a directory, use <dir>/metrics.jsonl; if a file, use directly.
      - Load the first record from each JSONL via dch_pipeline.determinism.load_first_jsonl_record().
      - If two or more records:
          Compare record[0] vs record[k] for k in [1..] using compare_artifacts(tol=tol).
          Collect: {"i": 0, "j": k, "equal": bool, "diff": ...}
      - Ensure bundle_dir exists.
      - Write concatenated JSONL at <bundle_dir>/<concat_name> preserving input order.
      - Write manifest JSON at <bundle_dir>/<manifest_name> using make_manifest().

    Returns:
      {
        "bundle_dir": str,
        "concat": str,     # path to concatenated JSONL
        "manifest": str,   # path to manifest JSON
        "comparisons": list[dict],
      }
    """
    # 1) Resolve and load
    in_paths: List[str] = [ _resolve_path(p) for p in artifacts ]
    records: List[Mapping[str, Any]] = []
    for path in in_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact not found: {path}")
        rec = load_first_jsonl_record(path)
        records.append(rec)

    # 2) Determinism comparisons (0 vs k)
    comparisons: List[Dict[str, Any]] = []
    if len(records) >= 2:
        base = records[0]
        for j in range(1, len(records)):
            comp = compare_artifacts(base, records[j], tol=float(tol))
            comparisons.append({"i": 0, "j": j, "equal": bool(comp.get("equal")), "diff": comp.get("diff")})

    # 3) Prepare output paths
    os.makedirs(bundle_dir, exist_ok=True)
    concat_path = os.path.join(bundle_dir, concat_name)
    manifest_path = os.path.join(bundle_dir, manifest_name)

    # 4) Write concatenated JSONL (preserve order)
    with open(concat_path, "w", encoding="utf-8") as f_out:
        for rec in records:
            f_out.write(json.dumps(rec, sort_keys=True) + "\n")

    # 5) Write manifest
    manifest = make_manifest(records, comparisons=comparisons, label=label)
    with open(manifest_path, "w", encoding="utf-8") as f_m:
        f_m.write(json.dumps(manifest, sort_keys=True) + "\n")

    return {
        "bundle_dir": bundle_dir,
        "concat": concat_path,
        "manifest": manifest_path,
        "comparisons": comparisons,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create a portable results bundle and verify determinism.")
    p.add_argument(
        "--artifact",
        "-a",
        action="append",
        default=[],
        help="Path to artifact file or directory (metrics.jsonl). Provide multiple times for comparisons.",
    )
    p.add_argument("--bundle-dir", default="artifacts/bundle", help="Output directory for the bundle.")
    p.add_argument("--tol", type=float, default=0.0, help="Absolute tolerance for numeric comparisons.")
    p.add_argument("--label", type=str, default=None, help="Optional label for the manifest.")
    p.add_argument("--concat-name", type=str, default="artifacts.jsonl", help="Filename for concatenated JSONL.")
    p.add_argument("--manifest-name", type=str, default="manifest.json", help="Filename for manifest JSON.")
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if not args.artifact:
        parser.error("At least one --artifact is required (two or more recommended for comparisons).")

    result = run_main(
        artifacts=list(args.artifact),
        bundle_dir=str(args.bundle_dir),
        tol=float(args.tol),
        label=args.label,
        concat_name=str(args.concat_name),
        manifest_name=str(args.manifest_name),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()