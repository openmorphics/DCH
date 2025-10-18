import json
import importlib.util
from pathlib import Path
from typing import Callable, Any, Dict


def _load_run_main() -> Callable[..., Dict[str, Any]]:
    """
    Try to import run_main from scripts.run_gate_review (package-style).
    Fallback to loading the file directly to avoid package import requirements.
    """
    try:
        from scripts.run_gate_review import run_main  # type: ignore
        return run_main  # type: ignore[return-value]
    except Exception:
        repo_root = Path(__file__).resolve().parents[1]
        mod_path = repo_root / "scripts" / "run_gate_review.py"
        spec = importlib.util.spec_from_file_location("run_gate_review", mod_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load run_gate_review module")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
        run_main = getattr(mod, "run_main", None)
        if not callable(run_main):
            raise RuntimeError("run_main not found in run_gate_review module")
        return run_main  # type: ignore[return-value]


def test_gate_review_smoke(tmp_path: Path):
    run_main = _load_run_main()
    out_dir = tmp_path / "gate_out"
    res = run_main(output_dir=str(out_dir))  # type: ignore[misc]

    # Result shape
    assert isinstance(res, dict)
    assert "artifact" in res and "report" in res and "passed" in res

    report = res["report"]
    assert isinstance(report, dict)

    # Flags must be default-off
    flags = report.get("flags", {})
    assert flags.get("manifold_default_off") is True
    assert flags.get("dual_proof_default_off") is True
    assert flags.get("dpo_default_off") is True

    # CLIs presence
    clis = report.get("clis", {})
    required_cli_paths = [
        "scripts/run_quick_experiment.py",
        "scripts/run_dataset_experiment.py",
        "scripts/run_ablation_suite.py",
        "scripts/run_baselines_experiment.py",
        "scripts/run_stats_report.py",
        "scripts/make_results_bundle.py",
    ]
    for p in required_cli_paths:
        assert p in clis, f"Missing CLI entry {p}"
        assert clis[p] is True, f"CLI path check failed: {p}"

    # Docs presence
    docs = report.get("docs", {})
    required_docs = [
        "docs/AlgorithmSpecs.md",
        "docs/RESULTS.md",
        "docs/REPRODUCIBILITY.md",
        "docs/BASELINES.md",
        "docs/ReleaseNotes.md",
    ]
    for p in required_docs:
        assert p in docs, f"Missing docs entry {p}"
        assert docs[p] is True, f"Docs content check failed: {p}"

    # Tests presence
    tests = report.get("tests", {})
    required_tests = [
        "tests/test_dual_proof_gating.py",
        "tests/test_dpo_rules.py",
        "tests/test_dpo_confluence.py",
        "tests/test_results_bundle.py",
        "tests/test_run_dataset_experiment_cli.py",
        "tests/test_stats_report.py",
    ]
    for p in required_tests:
        assert p in tests, f"Missing tests entry {p}"
        assert tests[p] is True, f"Tests path check failed: {p}"

    # All checks must pass
    assert report.get("passed") is True


def test_gate_report_written(tmp_path: Path):
    run_main = _load_run_main()
    out_file = "gate_report.json"
    res = run_main(output_dir=str(tmp_path), report_name=out_file)  # type: ignore[misc]

    artifact = Path(res["artifact"])
    assert artifact.exists(), "Gate report file was not written"

    # The file content is the inner report dict
    on_disk = json.loads(artifact.read_text(encoding="utf-8"))
    assert on_disk == res["report"], "Report on disk differs from returned structure"