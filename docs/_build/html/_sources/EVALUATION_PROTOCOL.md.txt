# Standardized Evaluation Protocol — Dynamic Causal Hypergraph (DCH)

Status: Draft v0.1  
Date: 2025-10-04  
Owners: DCH Maintainers  
License: MIT

Scope
- Defines the end-to-end protocol for evaluating DCH on event-based vision benchmarks:
  - DVS Gesture (camera gestures)
  - N-MNIST (neuromorphic MNIST)
- Covers dataset acquisition, preprocessing/encoding, backends, training/evaluation splits, metrics, ablations, statistical testing, random seed control, compute reporting, and artifact capture in support of reproducible, publication-grade experiments.

Related documents and components
- Framework decision and scope: [docs/FrameworkDecision.md](docs/FrameworkDecision.md)
- Interfaces and typed models: [docs/Interfaces.md](docs/Interfaces.md), [dch_core/interfaces.py](dch_core/interfaces.py)
- Algorithm specifications: [docs/AlgorithmSpecs.md](docs/AlgorithmSpecs.md)
- Paper packaging and artifact plan: [docs/PaperPackaging.md](docs/PaperPackaging.md)
- Packaging and envs: [pyproject.toml](pyproject.toml), [requirements.txt](requirements.txt), [environment.yml](environment.yml)
- Container and CI: [Dockerfile](Dockerfile), [scripts/with_docker.sh](scripts/with_docker.sh), [.github/workflows/ci.yml](.github/workflows/ci.yml)

Overview of evaluation design
- Objective: Assess whether DCH’s causal credit assignment and structural plasticity improve sample efficiency and interpretability under event-driven workloads, while maintaining competitive accuracy.
- Hypotheses:
  1) H1: DCH achieves non-inferior accuracy to strong SNN baselines on DVS Gesture and N-MNIST.
  2) H2: DCH with FSM+Abstraction reduces traversal compute (proxy for credit search cost) at comparable accuracy.
  3) H3: DCH provides auditability via hyperpath evidence and rule promotion logs that correlate with performance gains.

Datasets
- DVS Gesture
  - Source: Tonic dataset (gesture sequences). Licensing per Tonic terms.
  - Canonical splits: Use Tonic’s standard train/val/test (if provided). Otherwise, construct stratified splits (gesture labels) with subject awareness if metadata allows.
- N-MNIST
  - Source: Tonic dataset (converted from MNIST via saccades). Licensing per Tonic terms.
  - Canonical split: 60k train / 10k test.

Acquisition and verification
- Download script: [scripts/download_datasets.py](scripts/download_datasets.py) (to be implemented).
- Verify SHA256 checksums when available; log dataset version and source URI.
- Store under ./data with structure:
  - ./data/dvs_gesture/...
  - ./data/nmnist/...
- Loaders (to be implemented):
  - [dch_data/dvs_gesture.py](dch_data/dvs_gesture.py)
  - [dch_data/nmnist.py](dch_data/nmnist.py)

Preprocessing and encoding
- Preprocessing transforms (to be implemented): [dch_data/transforms.py](dch_data/transforms.py)
  - DVS Gesture:
    - Optional cropping/padding to standard resolution (e.g., 128x128 or dataset-native).
    - Normalization of event polarities.
    - Windowing/events slicing per time_bin (EncoderConfig).
  - N-MNIST:
    - Maintain saccadic event order.
    - Normalize events; preserve timestamps.
- Encoders (to be implemented): [dch_data/encoders.py](dch_data/encoders.py)
  - Time bin parameter (EncoderConfig.time_bin) (e.g., 1000 µs) – grid for ablations.
  - Output: time-major spike tensors (T, B, H, W, C) or (T, B, N) depending on model.
  - Metadata: valid sequence lengths.

Backends and model configurations
- Backends via BAL:
  - Norse primary (LIF/LI&F layers, configurable dt, thresholds).
  - BindsNET parity (classic STDP pipelines), for baseline comparability.
- Backend interfaces: [dch_snn/interface.py](dch_snn/interface.py)
- Norse model examples (to be implemented): [dch_snn/norse_models.py](dch_snn/norse_models.py)
  - Shallow and medium SNNs suitable for event encodings.
- Trainer interface will support CPU-first evaluation; CUDA optional.

Evaluation splits and procedures
- DVS Gesture
  - If canonical splits exist: Use train/val/test as provided.
  - Else: Stratified K-fold cross-validation (K=5) on subjects/gestures if metadata permits; otherwise stratify by labels only.
  - Repeat CV R=3 (total 15 folds). Ensure no subject leakage if applicable.
- N-MNIST
  - Standard train/test split.
  - Repeated holdout: Evaluate with 5 different random seeds (see Seeds).
- Cross-validation script (to be implemented): [scripts/make_splits.py](scripts/make_splits.py)
  - Writes split indices to ./splits/{dataset}/fold_{k}.json with seed provenance.

Hyperparameters and search
- Default configs under Hydra:
  - [configs/experiments/dvs_gesture.yaml](configs/experiments/dvs_gesture.yaml)
  - [configs/experiments/nmnist.yaml](configs/experiments/nmnist.yaml)
  - [configs/pipeline.yaml](configs/pipeline.yaml)
  - [configs/model/norse_lif.yaml](configs/model/norse_lif.yaml)
  - [configs/dch.yaml](configs/dch.yaml), [configs/fsm.yaml](configs/fsm.yaml), [configs/scaffolding.yaml](configs/scaffolding.yaml)
- Search strategy:
  - Small grid/random search over: Encoder time_bin; DHG k; combination_order_max; traversal beam_size; prune_threshold; FSM promotion thresholds.
  - Keep search budget modest; report wall-clock and total runs.
- Log all tried configs with metrics; export best as summary JSON/CSV.

Metrics
- Primary:
  - Accuracy (%), macro-F1 (DVS Gesture).
  - Accuracy (%) (N-MNIST).
- Secondary/diagnostic:
  - Traversal compute proxy: number of hyperpaths discovered per target event; average beam expansions; average degree encountered.
  - Edge dynamics: edges added/pruned per minute; reliability distribution histograms.
  - FSM throughput: hyperpaths/sec; promotions per hour; number of HOEs created.
  - Resource: wall-clock time, CPU/GPU utilization (if available).
- Logging utilities (to be implemented): [dch_pipeline/logging_utils.py](dch_pipeline/logging_utils.py)

Statistical testing and reporting
- Random seeds: 5 (at minimum) distinct seeds for each configuration.
- Report mean ± 95% confidence intervals (CI).
- Hypothesis tests:
  - Paired t-test (normality assumed across seeds) DCH vs. baseline.
  - Wilcoxon signed-rank test (distribution-free) as robustness check.
  - Multiple comparisons: Benjamini–Hochberg FDR control (q=0.05) when testing multiple configs.
- Effect sizes:
  - Cohen’s d (paired), and Cliff’s delta for nonparametric effect size.
- Implementation (to be implemented): [dch_pipeline/stats.py](dch_pipeline/stats.py)

Baselines
- Norse surrogate-gradient SNN: [baselines/norse_sg.py](baselines/norse_sg.py)
- BindsNET STDP: [baselines/bindsnet_stdp.py](baselines/bindsnet_stdp.py)
- Baseline documentation and configurations: [docs/BASELINES.md](docs/BASELINES.md) (to be authored)
- Ensure baselines use the same preprocessing/encoding pipeline where applicable.

Ablations
- Disable FSM (no promotions), disable HOE abstraction, disable scaffolding (no freeze).
- Vary TC-kNN k and window size; vary beam_size; vary prune_threshold and reliability clamps.
- Report impact on accuracy and traversal compute proxy.

Determinism and seeds
- Global seed control utility: [dch_pipeline/seeding.py](dch_pipeline/seeding.py)
  - Set Python, NumPy, PyTorch seeds.
  - Configure torch.backends.cudnn.deterministic=True, benchmark=False when requested.
- Log seeds and all relevant framework/library versions into an environment fingerprint file via [scripts/collect_env.py](scripts/collect_env.py).

Compute and environment reporting
- Provide CPU-only runs and optional CUDA runs (if available).
- Record:
  - CPU model, RAM, OS.
  - GPU model, driver, CUDA version (if any).
  - Python version, PyTorch/Norse/BindsNET/Tonic versions.
  - Wall-clock per epoch and total time per experiment.
- Export to: artifacts/<run_id>/env.json and artifacts/<run_id>/metrics.csv

Reproducibility artifacts
- For each run:
  - Save config snapshot: artifacts/<run_id>/config.yaml
  - Save checkpoints (if applicable): artifacts/<run_id>/ckpt/
  - Save TensorBoard logs: artifacts/<run_id>/tb/
  - Save raw predictions/probabilities for test sets to support re-analysis
- Archive final tables (CSV) under [docs/export/](docs/export/) with provenance (script hash, timestamp).

Command-line workflows (to be validated post-implementation)
- CPU (Norse backend):
  - python scripts/run_experiment.py experiment=dvs_gesture backend=norse device=cpu
  - python scripts/run_experiment.py experiment=nmnist backend=norse device=cpu
- CUDA (if available):
  - python scripts/run_experiment.py experiment=dvs_gesture backend=norse device=cuda:0
- BindsNET parity:
  - python scripts/run_experiment.py experiment=dvs_gesture backend=bindsnet device=cpu
- Cross-validation:
  - python scripts/run_experiment.py experiment=dvs_gesture +cv.folds=5 +cv.repeats=3 +seeds=[1,2,3,4,5]
- Ablation examples:
  - python scripts/run_experiment.py experiment=dvs_gesture dch.k=3 dch.combination_order_max=2 traversal.beam_size=8 dch.prune_threshold=0.05
  - python scripts/run_experiment.py experiment=dvs_gesture fsm.enabled=false abstraction.enabled=false

Acceptance criteria (publication readiness)
- Results reported with mean ± 95% CI, n ≥ 5 seeds per configuration.
- Paired statistical tests vs. baselines with FDR control; effect sizes included.
- All configs and seeds logged; scripts and env manifests validated.
- Serialized artifacts and env fingerprints archived and referenced in the paper.
- Protocol reproducible on CPU-only environment (container or conda) within reasonable wall-clock constraints.

Change control
- Any update to metrics or splits requires bumping an experiment version id and adding migration notes.
- Maintain protocol parity across backends to ensure fair comparisons.

End of protocol