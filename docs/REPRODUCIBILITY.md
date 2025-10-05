# Reproducibility Guide — Dynamic Causal Hypergraph (DCH)

Status: Draft v0.1  
Date: 2025-10-04  
Owners: DCH Maintainers  
License: MIT

Purpose
- This guide documents how to reproduce all experiments and artifacts for the DCH submission (NeurIPS 2025 target). It specifies environments, seeds, dataset handling, configuration management, logging, and archival requirements.

Authoritative references
- Decision record: [docs/FrameworkDecision.md](docs/FrameworkDecision.md)
- Packaging plan: [docs/PaperPackaging.md](docs/PaperPackaging.md)
- Interfaces: [docs/Interfaces.md](docs/Interfaces.md), [dch_core/interfaces.py](dch_core/interfaces.py)
- Algorithms: [docs/AlgorithmSpecs.md](docs/AlgorithmSpecs.md)
- Evaluation protocol: [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md)
- CI and containers: [.github/workflows/ci.yml](.github/workflows/ci.yml), [Dockerfile](Dockerfile), [scripts/with_docker.sh](scripts/with_docker.sh)
- Runtime/Dev env: [pyproject.toml](pyproject.toml), [requirements.txt](requirements.txt), [environment.yml](environment.yml)

Scope and claims
- We provide CPU-only reproducibility out of the box. CUDA is optional (containers provided).
- We fix all random seeds for each run and report mean ± 95% CI across multiple seeds/splits.
- We archive all artifacts (configs, metrics, environment fingerprints, checkpoints) with provenance.

Environments

1) Python virtualenv (CPU)
- Create and activate:
  - `python3 -m venv .venv && source .venv/bin/activate`
- Install:
  - `pip install -U pip wheel`
  - `pip install -e .`
  - Optionally dev tools: `pip install -r requirements.txt`
- Verify:
  - `python -c "import torch, norse; print(torch.__version__)"`

2) Conda (CPU)
- Create:
  - `conda env create -f environment.yml`
- Activate:
  - `conda activate dch`
- Verify:
  - `python -c "import torch, norse; print(torch.__version__)"`

3) Docker (CPU and CUDA)
- Build CPU:
  - `bash scripts/with_docker.sh build cpu`
- Run CPU shell:
  - `bash scripts/with_docker.sh run cpu -- bash`
- Build CUDA:
  - `bash scripts/with_docker.sh build cuda`
- Run CUDA shell:
  - `bash scripts/with_docker.sh run cuda -- bash`
- Notes:
  - CUDA workflow requires NVIDIA drivers and nvidia-container-toolkit.

Dataset acquisition and layout
- Use the dataset downloader (to be implemented) [scripts/download_datasets.py](scripts/download_datasets.py) or manual instructions in [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md).
- Directory structure:
  - `./data/dvs_gesture/...`
  - `./data/nmnist/...`
- Verify checksums when available and record dataset versions (URIs/hashes) in environment fingerprint.

Configuration management
- Hydra-based configuration files (to be added):
  - [configs/pipeline.yaml](configs/pipeline.yaml) core pipeline
  - [configs/dch.yaml](configs/dch.yaml) DCH hypergraph parameters
  - [configs/fsm.yaml](configs/fsm.yaml) FSM settings
  - [configs/scaffolding.yaml](configs/scaffolding.yaml) task-aware policies
  - [configs/model/norse_lif.yaml](configs/model/norse_lif.yaml) Norse model defaults
  - [configs/experiments/dvs_gesture.yaml](configs/experiments/dvs_gesture.yaml)
  - [configs/experiments/nmnist.yaml](configs/experiments/nmnist.yaml)
  - [configs/cv.yaml](configs/cv.yaml) cross-validation
  - [configs/hyperparams_sweep.yaml](configs/hyperparams_sweep.yaml) sweeps
- Each run saves an immutable config snapshot under `artifacts/<run_id>/config.yaml`.

Randomness and determinism
- Seeds are centrally controlled (to be implemented) in [dch_pipeline/seeding.py](dch_pipeline/seeding.py):
  - Python `random.seed`, NumPy `np.random.seed`, PyTorch `torch.manual_seed`.
  - Optional deterministic flags: `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`.
- For each experiment, use at least 5 distinct seeds and report mean ± 95% CI.
- Record seeds in `artifacts/<run_id>/env.json`.

Environment fingerprinting
- Collect environment metadata (to be implemented) via [scripts/collect_env.py](scripts/collect_env.py), including:
  - OS, CPU, RAM.
  - GPU name/driver/CUDA (if available).
  - Python, PyTorch, Norse, BindsNET, Tonic, and other key library versions.
  - Dataset versions and checksums (if available).
- Persist to `artifacts/<run_id>/env.json`.

Logging and artifacts
- Metrics: `artifacts/<run_id>/metrics.csv` (append rows per step/epoch).
- TensorBoard: `artifacts/<run_id>/tb/` (scalars, histograms).
- Checkpoints: `artifacts/<run_id>/ckpt/` with `state_dict` and metadata.
- Hypergraph snapshots (optional for debugging): `artifacts/<run_id>/hypergraph/` (JSON deltas or snapshots).
- FSM promotions and rules: `artifacts/<run_id>/fsm.jsonl` (one record per promotion).
- CLI entrypoint (to be implemented): [scripts/run_experiment.py](scripts/run_experiment.py).

Baseline comparability
- Baselines should reuse the same preprocessing/encoding pipeline where applicable:
  - Norse surrogate-gradient: [baselines/norse_sg.py](baselines/norse_sg.py)
  - BindsNET STDP: [baselines/bindsnet_stdp.py](baselines/bindsnet_stdp.py)
- Ensure batch sizes, number of epochs, and evaluation metrics align with DCH runs.
- Report baseline configs, seeds, and environment fingerprint.

Statistical testing and reporting
- Implement statistical procedures in [dch_pipeline/stats.py](dch_pipeline/stats.py) to compute:
  - Paired t-test, Wilcoxon signed-rank.
  - Effect sizes: Cohen’s d (paired), Cliff’s delta.
  - Multiple comparison control: Benjamini–Hochberg FDR (q=0.05).
- Report:
  - Means ± 95% CI across seeds (and folds for CV).
  - p-values and effect sizes for DCH vs baselines.
  - Number of runs, total compute time per configuration.

Compute fairness and budgets
- Report wall-clock, CPU/GPU utilization (if available), and hardware specs.
- Keep hyperparameter search budgets comparable across methods; document search grids.

Step-by-step reproduction (CPU quickstart)
1) Create environment:
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install -U pip wheel && pip install -e . && pip install -r requirements.txt`
2) Acquire datasets:
   - `python scripts/download_datasets.py --dataset dvs_gesture --root ./data` (TBD)
   - `python scripts/download_datasets.py --dataset nmnist --root ./data` (TBD)
3) Run a smoke test (after stubs are implemented):
   - `pytest -q`
4) Run an experiment (Norse CPU; placeholder until pipeline is implemented):
   - `python scripts/run_experiment.py experiment=dvs_gesture backend=norse device=cpu`
5) Check artifacts:
   - Inspect `artifacts/<run_id>/metrics.csv`, `tb/`, `config.yaml`, `env.json`.

Archival for submission
- Prepare a release under [docs/export/](docs/export/):
  - Source archive (matching tag).
  - Docker images sha256 digests and Dockerfiles.
  - Environment files (`requirements.txt`, `environment.yml`).
  - Final result tables (CSV/JSON) with generation scripts and commit hash.
  - Paper PDF and supplementary (notebooks, figures).
- Tag release v1.0.0 for camera-ready.

Governance and citation
- License: MIT (to be added) [LICENSE](LICENSE)
- Citation metadata: [CITATION.cff](CITATION.cff)
- Code of Conduct: [docs/CODE_OF_CONDUCT.md](docs/CODE_OF_CONDUCT.md)
- Contributing guide: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)

Known limitations and notes
- Some nondeterminism may persist on GPU even with deterministic flags; CPU runs provide the reference.
- Baselines may require slight deviations for fairness (document any departures).
- Large logs/snapshots can be disabled for speed; ensure the final runs used for the paper re-enable required logs.

Change control
- Any modification to evaluation or reproduction steps requires:
  - Updating this document with a changelog entry.
  - Bumping an experiment version id in configs and result tables.

Appendix A — Checklist
- [ ] Environment created and verified (Python/Conda or Docker)
- [ ] Datasets downloaded and checksums verified (if available)
- [ ] Seeds fixed and recorded
- [ ] Configs snapshot saved per run
- [ ] Metrics, TB logs, and env fingerprints archived
- [ ] Results table reproduced with CI log and artifact hashes
- [ ] Release tagged and artifacts published to docs/export/

End of guide