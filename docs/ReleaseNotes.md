# Release Notes

This document tracks releases, changes, and archival guidance for the Dynamic Causal Hypergraph (DCH) project.

---

## v1.0.0 (2025-10-11)

Summary of delivered features:
- Core DCH
  - Dynamic hypergraph construction (TC‑kNN) — [dch_core/dhg.py](../dch_core/dhg.py)
  - Constrained backward traversal for credit assignment — [dch_core/traversal.py](../dch_core/traversal.py)
  - Plasticity and pruning — [dch_core/plasticity.py](../dch_core/plasticity.py)
  - Streaming FSM — [dch_core/fsm.py](../dch_core/fsm.py)
  - Hierarchical abstraction (scaffold/hook) — [dch_core/abstraction.py](../dch_core/abstraction.py)
  - Task-aware scaffolding (policy hook) — [dch_core/scaffolding.py](../dch_core/scaffolding.py)
  - WL-style embedding — [dch_core/embeddings/wl.py](../dch_core/embeddings/wl.py)
- Pipeline and evaluation
  - Orchestrator and configs — [dch_pipeline/pipeline.py](../dch_pipeline/pipeline.py), [configs/](../configs)
  - Metrics/evaluation utils — [dch_pipeline/metrics.py](../dch_pipeline/metrics.py), [dch_pipeline/evaluation.py](../dch_pipeline/evaluation.py)
  - Stats utilities for significance testing — [dch_pipeline/stats.py](../dch_pipeline/stats.py)
  - Logging utilities and artifacts — [dch_pipeline/logging_utils.py](../dch_pipeline/logging_utils.py)
- Data layer
  - Encoders and transforms — [dch_data/encoders.py](../dch_data/encoders.py), [dch_data/transforms.py](../dch_data/transforms.py)
  - Dataset scaffolds — [dch_data/nmnist.py](../dch_data/nmnist.py), [dch_data/dvs_gesture.py](../dch_data/dvs_gesture.py)
- CLI and tests
  - CLI runner (torch-optional) — `dch-run` → [scripts/run_experiment.py](../scripts/run_experiment.py)
  - Unit/integration tests (CPU) — [tests/](../tests)
- Optional SNN integration
  - Norse models interface — [dch_snn/norse_models.py](../dch_snn/norse_models.py), [dch_snn/interface.py](../dch_snn/interface.py)
  - Torch-optional default; enable via `snn.enabled=true`
- Benchmarks
  - Traversal/pipeline JSON benchmarks — [benchmarks/benchmark_traversal.py](../benchmarks/benchmark_traversal.py), [benchmarks/benchmark_pipeline.py](../benchmarks/benchmark_pipeline.py)
- Documentation
  - Minimal Sphinx docsite (optional) — [docs/conf.py](conf.py), [docs/index.rst](index.rst)
  - Usage and troubleshooting guides — [docs/USAGE.md](USAGE.md), [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md)
  - Module responsibility matrix — [docs/ModuleResponsibilityMatrix.md](ModuleResponsibilityMatrix.md)
  - Hardware mapping appendix — [docs/HardwareAppendix.md](HardwareAppendix.md)
  - Results templates — [docs/RESULTS.md](RESULTS.md)
  - Supplementary materials scaffold — [docs/SUPPLEMENTARY.md](SUPPLEMENTARY.md)

Torch-optional policy:
- Core pipeline, tests, and CLI run without `torch`, `norse`, or `tonic`.
- SNN features activate only when explicitly requested (`snn.enabled=true`).

---

## Tagging and archival guidance

Tag the release:
```bash
git tag -a v1.0.0 -m "DCH v1.0.0: core pipeline, torch-optional CLI, benchmarks, docs"
git push origin v1.0.0
```

Export artifacts to [docs/export/](export/):
- Recommended exports (after experiments complete):
  - Assembled PDFs/HTML: specs and whitepaper
  - Evaluation protocol: [docs/EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)
  - Reproducibility guide: [docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md)
  - Results tables: [docs/RESULTS.md](RESULTS.md) (freeze the filled version)
  - Module responsibility matrix: [docs/ModuleResponsibilityMatrix.md](ModuleResponsibilityMatrix.md)
  - Hardware appendix: [docs/HardwareAppendix.md](HardwareAppendix.md)
  - Supplementary figures: place final figures under [docs/figures/](figures/)
- Helper scripts (optional):
  - HTML export: [scripts/export_html.py](../scripts/export_html.py)
  - PDF export: [scripts/export_pdf.sh](../scripts/export_pdf.sh)

Docsite build (optional):
```bash
python -m pip install -U sphinx myst-parser furo
make -C docs html
# or:
python -m sphinx -b html docs docs/_build/html
```

---

## Changelog

Unreleased
- Add optional adapters and datasets
- Expand unit test coverage for traversal and FSM
- Improve docsite with API references via autodoc

v1.0.0 — 2025-10-11
- Initial stable release with torch-optional pipeline, CLI runner, benchmarks, and minimal docsite.