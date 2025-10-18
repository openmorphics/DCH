# Module Responsibility Matrix — Dynamic Causal Hypergraph (DCH)

Status: Draft v0.1  
Date: 2025-10-04  
Owners: DCH Maintainers  
License: MIT

Scope
- Maps each major module to its purpose, inputs/outputs, key responsibilities, invariants, complexity notes, and verification artifacts.
- Aligns with the evaluation and reproducibility plans.

Core entities and interfaces
- File: [dch_core/interfaces.py](dch_core/interfaces.py)
  - Purpose: Single source of truth for typed data models and Protocols.
  - Exposes:
    - Entities: Event(), Vertex(), Hyperedge(), HypergraphSnapshot(), Hyperpath(), PlasticityState() — see [dch_core.interfaces()](dch_core/interfaces.py:1)
    - Protocols: HypergraphOps(), GraphConnectivity(), DHGConstructor(), TraversalEngine(), PlasticityEngine(), EmbeddingEngine(), FSMEngine(), AbstractionEngine(), ScaffoldingPolicy() — see [dch_core.interfaces()](dch_core/interfaces.py:1)
    - Helpers: make_vertex_id(), make_edge_id(), is_temporally_admissible() — see [dch_core.interfaces()](dch_core/interfaces.py:1)
  - Invariants: temporal windows respected; B-connectivity enforced at traversal layer; reliability ∈ [0,1] with clamps.
  - Verification: unit tests (TBD) for serialization and helper functions.

Hypergraph storage and connectivity
- File: [dch_core/hypergraph_mem.py](dch_core/hypergraph_mem.py)
  - Implementations: InMemoryHypergraph(), StaticGraphConnectivity() — see [dch_core.hypergraph_mem()](dch_core/hypergraph_mem.py:1)
  - Responsibilities:
    - Append-only vertex ingestion; window queries; adjacency and dedup for hyperedges.
    - Pruning based on reliability threshold; snapshot export.
  - Complexity: window_query O(N) in memory backend; acceptable for CPU experiments; future PMA index possible.
  - Verification: integration smoke test; future targeted unit tests for dedup/prune.

Dynamic Hypergraph Construction (DHG)
- File: [dch_core/dhg.py](dch_core/dhg.py)
  - Implementation: DefaultDHGConstructor() — see [dch_core.dhg.DefaultDHGConstructor()](dch_core/dhg.py:1)
  - Responsibilities:
    - TC-kNN candidate generation (per-presyn recent spikes) and higher-order tail enumeration under δ_causal.
    - Temporal admissibility checks and refractory guard; dedup and per-head budgets; admission into store.
  - Inputs: HypergraphOps, GraphConnectivity, head Vertex, window, params (k, m_max, δ_causal, budget, ρ).
  - Outputs: Candidate Hyperedges; admitted EdgeIds.
  - Complexity: O(d·k + |Cand| log |Cand| + bounded combinations); controllable by m_max and budgets.
  - Verification: smoke test; targeted unit tests for candidate enumeration (TBD).

Backward traversal and credit paths
- File: [dch_core/traversal.py](dch_core/traversal.py)
  - Implementation: DefaultTraversalEngine() — see [dch_core.traversal.DefaultTraversalEngine()](dch_core/traversal.py:1)
  - Responsibilities:
    - Beam search with AND-frontier; B-connectivity and temporal logic; horizon bound; reliability-composed scoring.
    - Canonical labeling for deduplication.
  - Inputs: HypergraphOps, target Vertex, horizon, beam_size.
  - Outputs: Sequence of Hyperpath() with scores and labels.
  - Complexity: ~O(D·B·b log(B·b)), memory O(D·B).
  - Verification: smoke test; targeted unit tests (TBD) for admissibility filters and scoring stability.

Plasticity and pruning
- File: [dch_core/plasticity.py](dch_core/plasticity.py)
  - Implementation: DefaultPlasticityEngine() — see [dch_core.plasticity.DefaultPlasticityEngine()](dch_core/plasticity.py:1)
  - Responsibilities:
    - Evidence aggregation over Hyperpath(), EMA update with clamping, counters, last_update_t.
    - Delegated pruning to HypergraphOps.
  - Inputs: Hyperpaths, sign (+1/-1), now_t, PlasticityState().
  - Outputs: Mapping[EdgeId -> ReliabilityScore]; prune count.
  - Complexity: O(sum |p|) + O(|E_active|); prune sweep O(|E|).
  - Verification: targeted unit tests (TBD) for potentiation/depression balance and clamping.

Streaming frequent hyperpath mining (FSM)
- File: [dch_core/fsm.py](dch_core/fsm.py)
  - Implementation: StreamingFSMEngine() — see [dch_core.fsm.StreamingFSMEngine()](dch_core/fsm.py:1)
  - Responsibilities:
    - Decayed counting of hyperpath labels; hysteresis-based promotions.
  - Inputs: Hyperpaths (with labels and scores), now_t; thresholds (θ), λ_decay, hold_k.
  - Outputs: Promoted label list.
  - Complexity: O(|P_t|·L) with O(1) per-label update.
  - Verification: targeted unit tests (TBD) for promotion stability under decay and noise.

Pipeline orchestration
- File: [dch_pipeline/pipeline.py](dch_pipeline/pipeline.py)
  - Configs: DHGConfig(), TraversalConfig(), PlasticityConfig(), PipelineConfig() — see [dch_pipeline.pipeline()](dch_pipeline/pipeline.py:1)
  - Implementation: DCHPipeline.step() — see [dch_pipeline.pipeline.DCHPipeline.step()](dch_pipeline/pipeline.py:97)
  - Responsibilities:
    - Ingest events; generate/admit DHG candidates; optional traversal/credit; plasticity updates; pruning; metrics.
    - Convenience constructor with in-memory store and defaults.
  - Inputs: Sequence[Event], optional targets (VertexId), signs, freeze flag.
  - Outputs: Metrics dict (candidates/admissions/hyperpaths/updates/pruned).
  - Verification: [tests/test_integration_smoke.py](tests/test_integration_smoke.py)

Seeding and determinism
- File: [dch_pipeline/seeding.py](dch_pipeline/seeding.py)
  - APIs: set_global_seeds(), enable_torch_determinism(), environment_seed_context() — see [dch_pipeline.seeding()](dch_pipeline/seeding.py:1)
  - Responsibilities:
    - Global seed setting; deterministic flags; context management for experiments.
  - Verification: import/runtime checks; used in runner.

SNN backend abstraction (BAL)
- File: [dch_snn/interface.py](dch_snn/interface.py)
  - Configs: EncoderConfig(), ModelConfig(), TrainerConfig() — see [dch_snn.interface()](dch_snn/interface.py:1)
  - Protocols: Encoder, SNNModel, Trainer, BackendAdapter — see [dch_snn.interface()](dch_snn/interface.py:1)
  - Responsibilities:
    - Contract for dual-backend adapters (Norse primary, BindsNET parity).
  - Verification: adapter-specific tests (TBD) once implementations land.

Encoders (data)
- File: [dch_data/encoders.py](dch_data/encoders.py)
  - Implementation: SimpleBinnerEncoder() — see [dch_data.encoders.SimpleBinnerEncoder()](dch_data/encoders.py:1)
  - Responsibilities:
    - Time-binning events to spike tensors; metadata for downstream consumers.
  - Verification: unit tests (TBD) for bin alignment and normalization.

Runner and artifacts
- File: [scripts/run_experiment.py](scripts/run_experiment.py)
  - Responsibilities:
    - CLI; seed setup; pipeline construction; synthetic events; metrics CSV; env/config JSON.
  - Verification: manual run and CI smoke once tests extended.

Stats utilities
- File: [dch_pipeline/stats.py](dch_pipeline/stats.py)
  - APIs: paired_ttest(), wilcoxon_signed_rank(), cohens_d_paired(), cliffs_delta(), benjamini_hochberg(), aggregate_runs() — see [dch_pipeline.stats()](dch_pipeline/stats.py:1)
  - Responsibilities:
    - Statistical testing and reporting for evaluation.
  - Verification: unit tests (TBD) against known small arrays.

Integration tests
- File: [tests/test_integration_smoke.py](tests/test_integration_smoke.py)
  - Responsibilities:
    - Minimal CPU smoke path: ingest → DHG → traversal → plasticity; metric sanity checks.
  - Next steps:
    - Add targeted unit tests for each engine and data module; add regression suite for traversal and DHG.

Planned modules (stubs pending)
- [dch_snn/registry.py](dch_snn/registry.py) — registry to select backend adapter by name; plugin entrypoints (optional).

Embeddings
- File: [dch_core/embeddings/wl.py](dch_core/embeddings/wl.py)
  - Purpose: Weisfeiler–Lehman-style hyperpath embedding for label features and similarity.
  - Verification: [tests/test_embeddings_wl.py](../tests/test_embeddings_wl.py)

Evaluation and metrics
- Files: [dch_pipeline/evaluation.py](dch_pipeline/evaluation.py), [dch_pipeline/metrics.py](dch_pipeline/metrics.py)
  - Responsibilities: streaming metrics (accuracy, macro/micro F1, confusion), evaluation helpers.
  - Verification: [tests/test_metrics_evaluation.py](../tests/test_metrics_evaluation.py)

Datasets and transforms
- Files: [dch_data/nmnist.py](dch_data/nmnist.py), [dch_data/dvs_gesture.py](dch_data/dvs_gesture.py), [dch_data/transforms.py](dch_data/transforms.py)
  - Responsibilities: dataset adapters (lazy tonic import), event transforms (torch-free).
  - Verification: [tests/test_data_nmnist.py](../tests/test_data_nmnist.py), [tests/test_data_dvs_gesture.py](../tests/test_data_dvs_gesture.py), [tests/test_utils_transforms_embedding.py](../tests/test_utils_transforms_embedding.py)

SNN backends (optional)
- File: [dch_snn/norse_models.py](dch_snn/norse_models.py)
  - Responsibilities: Norse reference models and trainer adapter (activated when `snn.enabled=true`).
  - Verification: future adapter tests (TBD); torch-optional policy enforced in runner.

Scripts and configs
- Scripts: [scripts/run_experiment.py](../scripts/run_experiment.py), [scripts/download_datasets.py](../scripts/download_datasets.py), [scripts/make_splits.py](../scripts/make_splits.py)
- Configs: [configs/pipeline.yaml](../configs/pipeline.yaml), [configs/dch.yaml](../configs/dch.yaml), [configs/fsm.yaml](../configs/fsm.yaml), [configs/scaffolding.yaml](../configs/scaffolding.yaml), [configs/experiments/dvs_gesture.yaml](../configs/experiments/dvs_gesture.yaml), [configs/experiments/nmnist.yaml](../configs/experiments/nmnist.yaml), [configs/model/norse_lif.yaml](../configs/model/norse_lif.yaml)

Benchmarks
- Files: [benchmarks/benchmark_traversal.py](../benchmarks/benchmark_traversal.py), [benchmarks/benchmark_pipeline.py](../benchmarks/benchmark_pipeline.py)
  - Responsibilities: deterministic performance probes printing single-line JSON.
  - Verification: manual run in CI and local; compare JSON keys against expected schema.

Traceability
- Formal algorithms and complexities: [docs/AlgorithmSpecs.md](docs/AlgorithmSpecs.md)
- Evaluation and reproducibility: [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md), [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
- Decision record: [docs/FrameworkDecision.md](docs/FrameworkDecision.md)

Acceptance
- Each module must provide minimal docstrings, respect Protocols, and include or reference tests.
- CI enforces lint/type/tests; containers provide CPU/CUDA reproducibility.
