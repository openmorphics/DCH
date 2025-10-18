from __future__ import annotations

import importlib


def test_version_string():
    from dch_core import __version__
    assert isinstance(__version__, str) and __version__.count(".") >= 1


def _assert_callable(module_path: str):
    mod = importlib.import_module(module_path)
    fn = getattr(mod, "run_main", None)
    assert callable(fn), f"run_main not callable in {module_path}"


def test_entrypoints_importability():
    modules = [
        "scripts.run_quick_experiment",
        "scripts.run_dataset_experiment",
        "scripts.run_ablation_suite",
        "scripts.run_baselines_experiment",
        "scripts.run_stats_report",
        "scripts.make_results_bundle",
        "scripts.run_gate_review",
    ]
    for m in modules:
        _assert_callable(m)