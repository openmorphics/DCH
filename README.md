# Dynamic Causal Hypergraph (DCH)
A neuro-symbolic causal framework for event-driven Spiking Neural Networks (SNNs)

Status: v0.1 (scaffold) • License: MIT • Target venue: NeurIPS 2025

DCH reframes SNN learning as continuous causal inference. It builds an evolving temporal hypergraph of spike events and causal hypotheses, assigns credit via constrained backward hyperpath traversal, and induces symbolic rules online via streaming frequent subgraph mining. This enables interpretable, auditable learning with explicit causal evidence trails.

Key features (publication checklist aligned)
- Typed data models and interfaces: Events, Hyperedges, Hypergraph, Hyperpaths, Plasticity state (see docs/Interfaces.md)
- Modular pipeline: preprocessing, encoding, dynamic hypergraph construction (TC‑kNN), traversal, credit assignment, plasticity, FSM, abstraction, evaluation
- Algorithm specs: pseudocode + complexity for hyperpath construction, credit assignment, adaptive policies (see docs/AlgorithmSpecs.md)
- Dual SNN backend via Backend Abstraction Layer (BAL): Norse primary, BindsNET parity (see dch_snn/interface.py)
- Reproducibility: complete protocol for DVS Gesture and N‑MNIST, CV and seed handling, stats and effect sizes, containers, CI (see docs/EVALUATION_PROTOCOL.md, docs/REPRODUCIBILITY.md)
- Baselines: Norse surrogate-gradient, BindsNET STDP (to be implemented)
- Hardware appendix (FPGA/neuromorphic mapping) (to be implemented)

Repository structure (in-progress)
- Core libraries
  - dch_core/: interfaces, DHG (TC‑kNN), traversal, plasticity, embeddings, FSM, abstraction, scaffolding
  - dch_snn/: backend-neutral interfaces, Norse/BindsNET adapters
  - dch_pipeline/: orchestration, evaluation, metrics, stats, seeding, logging
  - dch_data/: datasets (DVS Gesture, N‑MNIST), transforms, encoders
  - baselines/: Norse surrogate-gradient, BindsNET STDP
  - benchmarks/: pipeline and traversal microbenchmarks
- Tooling and packaging
  - pyproject.toml • requirements.txt • environment.yml • Dockerfile • .pre-commit-config.yaml • .github/workflows/ci.yml
- Documentation
  - docs/FrameworkDecision.md • docs/Interfaces.md • docs/AlgorithmSpecs.md • docs/EVALUATION_PROTOCOL.md • docs/REPRODUCIBILITY.md
  - docs/BASELINES.md (TBD) • docs/HardwareAppendix.md (TBD) • docs/API_REFERENCE.md (TBD)
- Scripts
  - scripts/run_experiment.py (TBD) • scripts/make_splits.py (TBD) • scripts/download_datasets.py (TBD)
  - scripts/collect_env.py (TBD) • scripts/with_docker.sh

Quickstart (CPU, Python venv)
1) Create environment
- python3 -m venv .venv && source .venv/bin/activate
- pip install -U pip wheel
- pip install -e .
- pip install -r requirements.txt

2) Run checks (once tests exist)
- pytest -q

3) Reproduce a CPU experiment (after pipeline land)
- python scripts/run_experiment.py experiment=dvs_gesture backend=norse device=cpu
- python scripts/run_experiment.py experiment=nmnist backend=norse device=cpu

Conda (CPU)
- conda env create -f environment.yml
- conda activate dch

Docker
- Build CPU: bash scripts/with_docker.sh build cpu
- Run CPU shell: bash scripts/with_docker.sh run cpu -- bash
- Build CUDA: bash scripts/with_docker.sh build cuda
- Run CUDA shell: bash scripts/with_docker.sh run cuda -- bash

Datasets
- DVS Gesture, N‑MNIST via Tonic. Use scripts/download_datasets.py (TBD) or follow docs/EVALUATION_PROTOCOL.md. Default data root: ./data

Reproducibility and evaluation
- Protocol: docs/EVALUATION_PROTOCOL.md
- Reproducibility guide: docs/REPRODUCIBILITY.md
- Statistical testing: dch_pipeline/stats.py (paired t‑test, Wilcoxon, FDR, effect sizes)
- Artifacts per run:
  - artifacts/<run_id>/config.yaml • metrics.csv • tb/ (TensorBoard) • env.json • ckpt/

Backend Abstraction Layer (BAL)
- Interface: dch_snn/interface.py
- Adapters:
  - Norse (primary): LIF/LI&F models, CPU+CUDA
  - BindsNET (parity): classic STDP baselines
- DCH traversal/credit does not depend on the SNN backend internals beyond emitted spikes/outputs.

Roadmap and status (high-level)
- M0: Packaging, CI, containers, interfaces, evaluation protocol [DONE/IN PROGRESS]
- M1: Data loaders, encoders, pipeline runner [PENDING]
- M2: DCH core (DHG, traversal, plasticity, embeddings, FSM, abstraction, scaffolding) [PENDING]
- M3: Baselines, metrics, CV/stats, result tables [PENDING]
- M4: API docs, examples, notebooks, hardware appendix [PENDING]
- M5: v1.0.0 release, archival under docs/export/ [PENDING]

Contributing
- Contributions welcome after v0.1 interfaces are stabilized.
- Please run pre-commit locally before pushing:
  - pip install pre-commit && pre-commit install && pre-commit run --all-files
- See docs/CONTRIBUTING.md (TBD) and docs/CODE_OF_CONDUCT.md (TBD)

Citing DCH
- Citation metadata will be provided in CITATION.cff upon first preprint.
- For now, please cite the repository and technical specification (docs/ and export artifacts).

License
- MIT. See LICENSE (TBD).

Acknowledgements
- Built with PyTorch, Norse, Tonic, and SciPy/NumPy; thanks to maintainers and community.

Contact
- Please file issues and discussions in this repository once public. For private feedback, refer to the corresponding author listed in CITATION.cff (TBD).