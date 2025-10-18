# Paper Packaging and Reproducibility Plan for DCH (NeurIPS 2025)

Status: Draft v0.1
Date: 2025-10-04
Owners: DCH Maintainers

Decision baseline
- Framework and targets: see [docs/FrameworkDecision.md](docs/FrameworkDecision.md)
- License: MIT
- Venue: NeurIPS 2025 main track with Code/Artifacts submission and Reproducibility Checklist

Goals and scope
- Deliver a publication-ready Dynamic Causal Hypergraph (DCH) implementation with complete artifacts for peer review and post-publication reuse.
- Provide deterministic, documented experiment pipelines for DVS Gesture and N-MNIST with dual SNN backends (Norse primary, BindsNET parity).
- Ship standardized baselines, cross-validation, statistical testing, and environment reproducibility.

Repository structure (authoritative)
- Core
  - [pyproject.toml](pyproject.toml) project metadata and dependencies
  - [requirements.txt](requirements.txt) pip lock for CPU
  - [environment.yml](environment.yml) conda env for CPU
  - [Dockerfile](Dockerfile) multi-stage CPU and CUDA images
  - [.github/workflows/ci.yml](.github/workflows/ci.yml) CI matrix
  - [.pre-commit-config.yaml](.pre-commit-config.yaml) formatting, linting, license headers
  - [README.md](README.md) overview and quickstart
  - [LICENSE](LICENSE) MIT
  - [CITATION.cff](CITATION.cff) citation metadata
- Data and preprocessing
  - [dch_data/dvs_gesture.py](dch_data/dvs_gesture.py)
  - [dch_data/nmnist.py](dch_data/nmnist.py)
  - [dch_data/transforms.py](dch_data/transforms.py)
  - [dch_data/encoders.py](dch_data/encoders.py)
  - [scripts/download_datasets.py](scripts/download_datasets.py)
- SNN backend integration
  - [dch_snn/interface.py](dch_snn/interface.py) backend-neutral contracts
  - [dch_snn/registry.py](dch_snn/registry.py) backend registry
  - [dch_snn/norse_models.py](dch_snn/norse_models.py)
  - [dch_snn/bindsnet_adapter.py](dch_snn/bindsnet_adapter.py)
- DCH core
  - [dch_core/interfaces.py](dch_core/interfaces.py) typed models
  - [dch_core/events.py](dch_core/events.py)
  - [dch_core/dhg.py](dch_core/dhg.py) dynamic hypergraph construction (TC-kNN)
  - [dch_core/traversal.py](dch_core/traversal.py) constrained backward credit
  - [dch_core/plasticity.py](dch_core/plasticity.py)
  - [dch_core/embeddings/wl.py](dch_core/embeddings/wl.py)
  - [dch_core/fsm.py](dch_core/fsm.py)
  - [dch_core/abstraction.py](dch_core/abstraction.py)
  - [dch_core/scaffolding.py](dch_core/scaffolding.py)
- Pipeline and evaluation
  - [dch_pipeline/pipeline.py](dch_pipeline/pipeline.py)
  - [dch_pipeline/evaluation.py](dch_pipeline/evaluation.py)
  - [dch_pipeline/metrics.py](dch_pipeline/metrics.py)
  - [dch_pipeline/stats.py](dch_pipeline/stats.py)
  - [dch_pipeline/seeding.py](dch_pipeline/seeding.py)
  - [dch_pipeline/logging_utils.py](dch_pipeline/logging_utils.py)
  - [scripts/run_experiment.py](scripts/run_experiment.py)
  - [scripts/make_splits.py](scripts/make_splits.py)
  - [scripts/collect_env.py](scripts/collect_env.py)
  - [scripts/with_docker.sh](scripts/with_docker.sh)
- Baselines
  - [baselines/norse_sg.py](baselines/norse_sg.py)
  - [baselines/bindsnet_stdp.py](baselines/bindsnet_stdp.py)
  - [docs/BASELINES.md](docs/BASELINES.md)
- Configs (Hydra)
  - [configs/pipeline.yaml](configs/pipeline.yaml)
  - [configs/dch.yaml](configs/dch.yaml)
  - [configs/fsm.yaml](configs/fsm.yaml)
  - [configs/scaffolding.yaml](configs/scaffolding.yaml)
  - [configs/model/norse_lif.yaml](configs/model/norse_lif.yaml)
  - [configs/experiments/dvs_gesture.yaml](configs/experiments/dvs_gesture.yaml)
  - [configs/experiments/nmnist.yaml](configs/experiments/nmnist.yaml)
  - [configs/cv.yaml](configs/cv.yaml)
  - [configs/hyperparams_sweep.yaml](configs/hyperparams_sweep.yaml)
- Tests and benchmarks
  - [tests/test_data_dvs_gesture.py](tests/test_data_dvs_gesture.py)
  - [tests/test_data_nmnist.py](tests/test_data_nmnist.py)
  - [tests/test_encoders.py](tests/test_encoders.py)
  - [tests/test_dhg_tc_knn.py](tests/test_dhg_tc_knn.py)
  - [tests/test_traversal_credit.py](tests/test_traversal_credit.py)
  - [tests/test_plasticity_rules.py](tests/test_plasticity_rules.py)
  - [tests/test_fsm_streaming.py](tests/test_fsm_streaming.py)
  - [tests/test_abstraction_ho_edges.py](tests/test_abstraction_ho_edges.py)
  - [tests/test_scaffolding_policy.py](tests/test_scaffolding_policy.py)
  - [tests/test_snn_integration.py](tests/test_snn_integration.py)
  - [tests/test_integration_smoke.py](tests/test_integration_smoke.py)
  - [benchmarks/benchmark_pipeline.py](benchmarks/benchmark_pipeline.py)
  - [benchmarks/benchmark_traversal.py](benchmarks/benchmark_traversal.py)
- Documentation and artifacts
  - [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
  - [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md)
  - [docs/USAGE.md](docs/USAGE.md)
  - [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
  - [docs/ModuleResponsibilityMatrix.md](docs/ModuleResponsibilityMatrix.md)
  - [docs/AlgorithmSpecs.md](docs/AlgorithmSpecs.md)
  - [docs/HardwareAppendix.md](docs/HardwareAppendix.md)
  - [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
  - [docs/SUPPLEMENTARY.md](docs/SUPPLEMENTARY.md)
  - [docs/RESULTS.md](docs/RESULTS.md)
  - [docs/export/](docs/export/) produced PDFs, HTML, CSVs

NeurIPS Reproducibility Checklist mapping (abbrev)
- Data: dataset sources, licenses, preprocessing documented in [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md)
- Code: full training/credit assignment, evaluation, and baselines included
- Hyperparameters: all defaults and sweeps in [configs/](configs/)
- Compute: hardware, time, and energy estimates reported in [docs/RESULTS.md](docs/RESULTS.md)
- Randomness: seeds and determinism options in [dch_pipeline/seeding.py](dch_pipeline/seeding.py)
- Results: average ± 95% CI; raw runs archived under [docs/export/](docs/export/)

Quickstart (CPU)
1) Python with pip
- Install: `python3 -m venv .venv && source .venv/bin/activate`
- Upgrade tools: `pip install -U pip wheel`
- Install project: `pip install -e .`
- Optional developer tools: `pip install -r requirements.txt`

2) Conda
- Create env: `conda env create -f environment.yml`
- Activate: `conda activate dch`

3) Docker (CPU)
- Build: `docker build -t dch:cpu -f Dockerfile --target dch_cpu .`
- Run: `bash scripts/with_docker.sh cpu`

CUDA with Docker (optional)
- Build: `docker build -t dch:cuda -f Dockerfile --target dch_cuda .`
- Run: `bash scripts/with_docker.sh cuda`

Dataset acquisition
- DVS Gesture
  - Script: `python scripts/download_datasets.py --dataset dvs_gesture --root ./data`
  - Loader: [dch_data/dvs_gesture.py](dch_data/dvs_gesture.py)
- N-MNIST
  - Script: `python scripts/download_datasets.py --dataset nmnist --root ./data`
  - Loader: [dch_data/nmnist.py](dch_data/nmnist.py)
- Notes
  - Verify checksums; respect dataset licenses.
  - First run downloads may be cached by backends (e.g., Tonic).
  - For air-gapped environments, see manual instructions in [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md).

Running experiments (Hydra)
- CPU default, Norse backend:
  - `python scripts/run_experiment.py experiment=dvs_gesture backend=norse device=cpu`
  - `python scripts/run_experiment.py experiment=nmnist backend=norse device=cpu`
- BindsNET parity:
  - `python scripts/run_experiment.py experiment=dvs_gesture backend=bindsnet device=cpu`
- CUDA (if available):
  - `python scripts/run_experiment.py experiment=dvs_gesture backend=norse device=cuda:0`
- Cross-validation and seeds:
  - `python scripts/run_experiment.py +cv.folds=5 +cv.repeats=3 +seeds=[1,2,3]`

Outputs and logging
- Metrics CSV: `artifacts/<run_id>/metrics.csv`
- TensorBoard: `artifacts/<run_id>/tb/`
- Config snapshot: `artifacts/<run_id>/config.yaml`
- Checkpoints: `artifacts/<run_id>/ckpt/`
- Environment fingerprint: `artifacts/<run_id>/env.json` (from [scripts/collect_env.py](scripts/collect_env.py))

Evaluation protocol (summary)
- Datasets
  - DVS Gesture: standard splits; event-to-spike encoding per config; report accuracy and F1.
  - N-MNIST: standard train/test; report accuracy.
- Baselines
  - Norse surrogate-gradient SNN [baselines/norse_sg.py](baselines/norse_sg.py)
  - BindsNET STDP [baselines/bindsnet_stdp.py](baselines/bindsnet_stdp.py)
- DCH runs
  - Dual-backend compatible encoders; DCH credit assignment enabled.
  - Ablations: disable FSM, disable abstraction, vary TC-kNN k and window.
- Statistics
  - Repeated CV or repeated holdout, at least 5 seeds.
  - Tests: paired t-test and Wilcoxon signed-rank; report effect sizes (Cohen’s d, Cliff’s delta).
  - Confidence intervals via bootstrap or t-intervals; multiple-comparison control per protocol.
- Compute budget
  - Report wall-clock and energy proxy; set maximum GPU hours where applicable.

Determinism and seeds
- Central seeding in [dch_pipeline/seeding.py](dch_pipeline/seeding.py): Python, NumPy, PyTorch, CuDNN flags (deterministic, benchmark off).
- Log all seeds, torch/CuDNN settings, and library versions to env.json.

Module responsibility matrix (pointer)
- See [docs/ModuleResponsibilityMatrix.md](docs/ModuleResponsibilityMatrix.md) for ownership, inputs/outputs, and invariants across:
  - Data loaders, encoders, SNN adapters, DCH core (DHG, traversal, plasticity, FSM, abstraction, scaffolding), pipeline, evaluation, stats, and logging.

Algorithm specifications (pointer)
- See [docs/AlgorithmSpecs.md](docs/AlgorithmSpecs.md) for pseudocode and complexity analysis:
  - Hyperpath construction and embedding
  - Temporal credit assignment via constrained backward traversal
  - Plasticity updates and adaptive thresholding
  - Streaming frequent hyperpath mining and promotion to rules

Hardware mapping (pointer)
- See [docs/HardwareAppendix.md](docs/HardwareAppendix.md) for FPGA/neuromorphic deployment considerations (no implementation).

Governance and contributor guidelines
- Code style: ruff + black, mypy type checking. Configure via [pyproject.toml](pyproject.toml).
- Pre-commit hooks: [.pre-commit-config.yaml](.pre-commit-config.yaml)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Citation metadata: [CITATION.cff](CITATION.cff)

Archival and versioning
- Tag release v1.0.0 upon acceptance. Provide:
  - Source tarball, Docker images, environment files, results CSV/JSON under [docs/export/](docs/export/)
  - Final PDF and supplementary (notebooks, figures) under [docs/export/](docs/export/)
- Semantic versioning for post-publication updates (v1.0.x bugfix, v1.y feature).

Submission checklist (condensed)
- [ ] Code builds and tests pass locally and in CI
- [ ] Reproducibility docs complete and validated
- [ ] Baselines run and results reproduced with logged configs and seeds
- [ ] Statistical tests computed with reported confidence intervals
- [ ] Environment manifests and Docker images verified
- [ ] Paper figures and tables exported from scripts with provenance
- [ ] License and citation files present and validated

Appendix A: Example commands (to be validated post-implementation)
- CPU DVS Gesture Norse:
  - `python scripts/run_experiment.py experiment=dvs_gesture backend=norse device=cpu trainer.max_epochs=50`
- CUDA DVS Gesture Norse:
  - `python scripts/run_experiment.py experiment=dvs_gesture backend=norse device=cuda:0 trainer.max_epochs=50`
- Baseline BindsNET STDP:
  - `python scripts/run_experiment.py experiment=nmnist backend=bindsnet device=cpu model=bindsnet_stdp`
- Benchmarks:
  - `python benchmarks/benchmark_traversal.py`
  - `python benchmarks/benchmark_pipeline.py`

Notes
- All file paths are authoritative targets; missing files will be created as part of the implementation phase following this plan.