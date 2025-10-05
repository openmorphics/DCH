# Framework Decision Record: DCH SNN Backends, Toolchain, and Publication Targets

Status: Accepted
Date: 2025-10-04
Decision ID: FDR-001
Owners: DCH Maintainers

Summary
- Primary backend: Norse (with parity via BindsNET adapter)
- Python: 3.10–3.12
- PyTorch: &#62;= 2.2
- Compute: CPU + CUDA (12.x); ROCm later if demand
- Publication target: NeurIPS 2025 (main) with Reproducibility Checklist
- License: MIT
- First deliverables: [docs/FrameworkDecision.md](docs/FrameworkDecision.md) and interface contracts; then scaffold repository
- CI: GitHub Actions matrix across OS/Python/PyTorch; CPU default and gated CUDA

Scope
This record fixes the backend architecture, compatibility targets, CI support, licensing, and publication packaging plan for the Dynamic Causal Hypergraph (DCH) implementation.

Rationale and trade-offs
- Norse provides production-grade SNN modules, integration with Tonic datasets, and active maintenance.
- BindsNET offers classic STDP pipelines; an adapter preserves baseline comparability and community reach.
- Dual-backend reduces framework lock-in and strengthens reproducibility across ecosystems.
- Python 3.10–3.12 aligns with PyTorch stable support; PyTorch &#62;= 2.2 enables modern kernels and performance.
- CPU-first CI ensures determinism; CUDA adds performance; ROCm is postponed to avoid matrix bloat initially.
- MIT license is widely adopted for academic code and compatible with dependencies.
- NeurIPS 2025 timeline supports full artifact preparation on DVS Gesture and N-MNIST.

Architecture decision: Backend Abstraction Layer (BAL)
We introduce a narrow BAL exposing common contracts for model construction, encoding, and training. Adapters implement these contracts for each backend.

Core contracts (modules to implement and stabilize)
- [dch_core/interfaces.py](dch_core/interfaces.py): typed data models (events, hyperedges, hypergraph, plasticity state)
- [dch_snn/interface.py](dch_snn/interface.py): backend-neutral SNN interfaces (SNNModel, Encoder, Trainer)
- [dch_snn/registry.py](dch_snn/registry.py): backend registry (backend string -&#62; adapter)

Backend adapters
- NorseAdapter: binds BAL to Norse primitives and training loop
- BindsNETAdapter: binds BAL to BindsNET modules; used for baseline parity

Adapter design notes
- Training loop heterogeneity is encapsulated behind Trainer; DCH credit assignment is framework-agnostic.
- Spike encoders wrap events into tensors/streams; both backends conform to Encoder.
- Checkpoint IO normalized via a common schema (state dict keys and metadata).
- Determinism helpers (seeding, CuDNN flags) applied centrally before invoking backends.

CI and tooling
- Matrix
  - OS: ubuntu-latest, macos-latest
  - Python: 3.10, 3.11, 3.12
  - Torch: pinned minor versions matching wheels
- Jobs
  - Lint/type: ruff, black, mypy
  - Unit/integration tests (CPU)
  - Optional CUDA tests on self-hosted or scheduled
- Caching: pip + torch wheels
- Artifacts: coverage XML, benchmark CSVs
- Key files: [pyproject.toml](pyproject.toml), [Dockerfile](Dockerfile), [.github/workflows/ci.yml](.github/workflows/ci.yml), [.pre-commit-config.yaml](.pre-commit-config.yaml)

Reproducibility and packaging
- Environment manifests: [pyproject.toml](pyproject.toml), [requirements.txt](requirements.txt), [environment.yml](environment.yml)
- Container: [Dockerfile](Dockerfile) with CUDA and CPU stages; helper script [scripts/with_docker.sh](scripts/with_docker.sh)
- Config: Hydra-based configs under [configs/](configs/)
- Seeds: global seed control and logging; environment fingerprint [scripts/collect_env.py](scripts/collect_env.py)
- Artifact capture: metrics CSV, TensorBoard, checkpoints, config snapshots
- Paper kit: [docs/PaperPackaging.md](docs/PaperPackaging.md) and [docs/USAGE.md](docs/USAGE.md) for end-to-end replication
- Governance: [CITATION.cff](CITATION.cff), [LICENSE](LICENSE), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), [CONTRIBUTING.md](CONTRIBUTING.md)

Publication plan
- Venue: NeurIPS 2025 (main)
- Tracks: code submission and reproducibility artifacts
- Reproducibility Checklist: answered in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
- Baselines: Norse surrogate-gradient and BindsNET STDP; protocol in [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md)

Risks and mitigations
- Dependency churn (Torch/CUDA): pin minor versions and use constraints; provide fallback CPU images
- Adapter drift: BAL contract tests and integration examples per backend
- Dataset availability: downloader with checksums; mirror links when permissible
- Performance variance: fixed seeds, repeated runs, effect sizes, confidence intervals
- GPU scarcity in CI: CPU-default tests; CUDA on schedule/self-hosted

Milestones
- M0: Create this record and finalize interfaces; scaffold repository
- M1: Data loaders, encoders, pipeline orchestration; CI and pre-commit green
- M2: DCH core (DHG, traversal, plasticity, FSM, abstraction, scaffolding)
- M3: Evaluation suite (datasets, baselines, metrics, stats, CV)
- M4: Paper artifacts (docsite, notebooks, results, ablations)
- M5: v1.0.0 release and archival into [docs/export/](docs/export/)

Acceptance criteria
- BAL interfaces finalized and covered by contract tests
- Dual-backend examples run end to end on CPU
- CI green across matrix; CUDA job passes on schedule
- Reproducibility docs complete and environment manifests validated
- Draft paper kit compiles with all artifacts present

Mermaid: plugin architecture
flowchart TD
    A[DCH Pipeline] --&#62; B[Backend Abstraction Layer]
    B --&#62; C[NorseAdapter]
    B --&#62; D[BindsNETAdapter]
    C --&#62; E[SNNModel]
    D --&#62; F[SNNModel]
    A --&#62; G[DCH Core]
    G --&#62; H[DHG]
    G --&#62; I[Traversal]
    G --&#62; J[Plasticity FSM Abstraction]

Related files
- [dch_core/interfaces.py](dch_core/interfaces.py)
- [dch_snn/interface.py](dch_snn/interface.py)
- [dch_snn/registry.py](dch_snn/registry.py)
- [pyproject.toml](pyproject.toml)
- [Dockerfile](Dockerfile)
- [.github/workflows/ci.yml](.github/workflows/ci.yml)