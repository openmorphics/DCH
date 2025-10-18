# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
P2-14 Gate review — approve research package for MVP API/library.

This opt-in CLI performs static checks only (no subprocesses, no network):
- Flags default-off (import-only):
  • manifold disabled by default (cfg.manifold is None or cfg.manifold.enable is False)
  • dual_proof disabled by default (cfg.dual_proof is None or cfg.dual_proof.enable is False)
  • dpo disabled by default (cfg.dpo is None or cfg.dpo.enable is False)
- CLIs present (path existence only) under scripts/
- Documentation sections present (simple substring scans)
- Representative tests present (path existence)

Output
- Writes JSON report to <output_dir>/<report_name>:
  {
    "flags": { ... },
    "clis": { ... },
    "docs": { ... },
    "tests": { ... },
    "passed": true|false
  }
- Prints a single-line JSON to stdout with:
  {"artifact": "<path>", "report": {…}, "passed": true|false}

Defaults policy
- All gating features must be OFF by default. This script validates the default-off
  behavior by importing PipelineConfig from dch_pipeline.pipeline and instantiating
  it with no arguments.

Intended usage
- Run in CI to produce a gate review artifact for dashboards:
    python -m scripts.run_gate_review --output-dir artifacts/gate --report-name gate_report.json
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _check_flags() -> Dict[str, bool]:
    """
    Import-only validation of default-off flags from dch_pipeline.pipeline.PipelineConfig.
    """
    try:
        from dch_pipeline.pipeline import PipelineConfig  # import-only
        cfg = PipelineConfig()
    except Exception:
        return {
            "manifold_default_off": False,
            "dual_proof_default_off": False,
            "dpo_default_off": False,
        }

    def _is_disabled(obj: object) -> bool:
        if obj is None:
            return True
        # Defensive: treat missing attribute as disabled
        return not bool(getattr(obj, "enable", False))

    return {
        "manifold_default_off": _is_disabled(getattr(cfg, "manifold", None)),
        "dual_proof_default_off": _is_disabled(getattr(cfg, "dual_proof", None)),
        "dpo_default_off": _is_disabled(getattr(cfg, "dpo", None)),
    }

def _check_paths(rel_paths: Mapping[str, str]) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for key, rel in rel_paths.items():
        out[rel] = (REPO_ROOT / rel).exists()
    return out

def _check_docs() -> Dict[str, bool]:
    docs: Dict[str, bool] = {}
    # AlgorithmSpecs: Dual-proof semantics, DPO, Beta–Bernoulli (dash tolerant)
    p = REPO_ROOT / "docs" / "AlgorithmSpecs.md"
    txt = _read_text(p)
    dp = bool(re.search(r"Dual[-– ]proof.*semantics", txt, flags=re.IGNORECASE | re.DOTALL))
    dpo = "DPO" in txt
    beta = bool(re.search(r"Beta(?:[-–]|[ ]?)[Bb]ernoulli", txt))
    docs[str(p.relative_to(REPO_ROOT))] = dp and dpo and beta

    # RESULTS: How to run + a CLI name
    p = REPO_ROOT / "docs" / "RESULTS.md"
    txt = _read_text(p)
    how = "how to run" in txt.lower()
    cli_names = ["run_quick_experiment", "run_dataset_experiment", "run_ablation_suite",
                 "run_baselines_experiment", "run_stats_report", "make_results_bundle"]
    any_cli = any(name in txt for name in cli_names)
    docs[str(p.relative_to(REPO_ROOT))] = how and any_cli

    # REPRODUCIBILITY: set_global_seeds and config_fingerprint
    p = REPO_ROOT / "docs" / "REPRODUCIBILITY.md"
    txt = _read_text(p)
    docs[str(p.relative_to(REPO_ROOT))] = ("set_global_seeds" in txt) and ("config_fingerprint" in txt)

    # BASELINES: baselines and dry-run
    p = REPO_ROOT / "docs" / "BASELINES.md"
    txt = _read_text(p)
    docs[str(p.relative_to(REPO_ROOT))] = ("baselines" in txt.lower()) and ("dry-run" in txt.lower())

    # ReleaseNotes: P2 Research Package
    p = REPO_ROOT / "docs" / "ReleaseNotes.md"
    txt = _read_text(p)
    docs[str(p.relative_to(REPO_ROOT))] = ("p2 research package".lower() in txt.lower())

    return docs

def run_main(output_dir: str = "artifacts/gate", report_name: str = "gate_report.json") -> Dict[str, object]:
    """
    Run gate review checks and emit a JSON report.

    Parameters
    - output_dir: directory to write the report artifact (created if missing)
    - report_name: filename for the JSON report
    """
    flags = _check_flags()
    clis = _check_paths({
        "quick": "scripts/run_quick_experiment.py",
        "dataset": "scripts/run_dataset_experiment.py",
        "ablation": "scripts/run_ablation_suite.py",
        "baselines": "scripts/run_baselines_experiment.py",
        "stats": "scripts/run_stats_report.py",
        "bundle": "scripts/make_results_bundle.py",
    })
    docs = _check_docs()
    tests = _check_paths({
        "dual_proof_gating": "tests/test_dual_proof_gating.py",
        "dpo_rules": "tests/test_dpo_rules.py",
        "dpo_confluence": "tests/test_dpo_confluence.py",
        "results_bundle": "tests/test_results_bundle.py",
        "run_dataset_experiment_cli": "tests/test_run_dataset_experiment_cli.py",
        "stats_report": "tests/test_stats_report.py",
    })

    report = {
        "flags": flags,
        "clis": clis,
        "docs": docs,
        "tests": tests,
        "passed": (all(flags.values()) and all(clis.values()) and all(docs.values()) and all(tests.values())),
    }

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    os.makedirs(out_dir, exist_ok=True)
    artifact_path = out_dir / report_name
    with artifact_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    result = {"artifact": str(artifact_path), "report": report, "passed": bool(report["passed"])}
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P2-14 Gate Review CLI (opt-in)")
    parser.add_argument("--output-dir", default="artifacts/gate")
    parser.add_argument("--report-name", default="gate_report.json")
    args = parser.parse_args()
    out = run_main(output_dir=args.output_dir, report_name=args.report_name)
    print(json.dumps(out, separators=(",", ":"), ensure_ascii=False))