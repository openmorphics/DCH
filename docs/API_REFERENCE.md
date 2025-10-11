# DCH API Reference

This reference summarizes the key modules, data models, and protocols that define the Dynamic Causal Hypergraph (DCH) framework. It focuses on stable, torch-free interfaces and core algorithms. Optional backends (e.g., Norse, BindsNET) integrate through a decoupled adapter layer.

Conventions
- Clickable source anchors reference the repository file and line for definitions, e.g., [`Event`](dch_core/interfaces.py:71).
- All time units (Timestamp) are integers (dataset-native or microseconds).
- Torch is optional; when absent, codepaths remain import-safe and tests stay green.

---

## Core Types and Entities

- Timestamp, NeuronId, VertexId, EdgeId, ReliabilityScore, Window  
  See type aliases in [`dch_core/interfaces.py`](dch_core/interfaces.py).

- Event  
  Structured event with neuron_id, t, and optional meta: [`Event`](dch_core/interfaces.py:71)

- Vertex  
  Materialized node for an event (id = "neuron@time"): [`Vertex`](dch_core/interfaces.py:82), [`make_vertex_id()`](dch_core/interfaces.py:57)

- Hyperedge  
  Directed, causal hypothesis from a non-empty tail set to a single head with temporal constraints and reliability: [`Hyperedge`](dch_core/interfaces.py:88)

- Hyperpath  
  DAG-like proof chain satisfying B-connectivity and temporal logic (with optional label for FSM): [`Hyperpath`](dch_core/interfaces.py:137)

- PlasticityState  
  Parameters for reliability updates and pruning: [`PlasticityState`](dch_core/interfaces.py:151)

- Helpers  
  [`make_edge_id()`](dch_core/interfaces.py:62), [`is_temporally_admissible()`](dch_core/interfaces.py:166)

---

## Core Protocols (Engine Interfaces)

These are the stable contracts your implementations must follow:

- Storage and indices: [`HypergraphOps`](dch_core/interfaces.py:181)
- Connectivity oracle: [`GraphConnectivity`](dch_core/interfaces.py:218)
- Dynamic Hypergraph Construction (TC‑kNN): [`DHGConstructor`](dch_core/interfaces.py:227)
- Constrained backward traversal: [`TraversalEngine`](dch_core/interfaces.py:259)
- Reliability updates and pruning: [`PlasticityEngine`](dch_core/interfaces.py:282)
- Hyperpath embeddings (WL-style, etc.): [`EmbeddingEngine`](dch_core/interfaces.py:310)
- Streaming FSM engine: [`FSMEngine`](dch_core/interfaces.py:321)
- Hierarchical abstraction (HOEs): [`AbstractionEngine`](dch_core/interfaces.py:336)
- Task-aware scaffolding policy: [`ScaffoldingPolicy`](dch_core/interfaces.py:353)

---

## Core Implementations

- In-memory hypergraph backend  
  [`dch_core/hypergraph_mem.py`](dch_core/hypergraph_mem.py)

- Dynamic Hypergraph Construction (TC‑kNN)  
  [`dch_core/dhg.py`](dch_core/dhg.py)

- Constrained backward traversal and scoring  
  [`dch_core/traversal.py`](dch_core/traversal.py)

- Plasticity updates and pruning  
  [`dch_core/plasticity.py`](dch_core/plasticity.py)

- WL-style hyperpath embedding (canonical labels and feature hashing)  
  [`dch_core/embeddings/wl.py`](dch_core/embeddings/wl.py)

- Streaming frequent hyperpath mining (decay + hysteresis; promotion queue)  
  [`dch_core/fsm.py`](dch_core/fsm.py)

- Hierarchical abstraction (promotion to higher‑order hyperedges)  
  [`dch_core/abstraction.py`](dch_core/abstraction.py)

- Task-aware scaffolding (FREEZE/REUSE/ISOLATE)  
  [`dch_core/scaffolding.py`](dch_core/scaffolding.py)

---

## Pipeline Orchestration and Utilities

- DCH Pipeline (ingestion → DHG → traversal → plasticity → FSM → abstraction)  
  [`dch_pipeline/pipeline.py`](dch_pipeline/pipeline.py)

- Metrics and evaluation (torch-free)  
  [`dch_pipeline/metrics.py`](dch_pipeline/metrics.py), [`dch_pipeline/evaluation.py`](dch_pipeline/evaluation.py)

- Deterministic seeding and environment fingerprint  
  [`dch_pipeline/seeding.py`](dch_pipeline/seeding.py)

- Logging (CSV, JSONL, TensorBoard; TB no-op if missing)  
  [`dch_pipeline/logging_utils.py`](dch_pipeline/logging_utils.py)

---

## Data, Transforms, and Encoders

- Event transforms (torch-free)  
  [`dch_data/transforms.py`](dch_data/transforms.py)

- Dataset loaders (lazy optional deps: tonic/numpy)  
  [`dch_data/nmnist.py`](dch_data/nmnist.py), [`dch_data/dvs_gesture.py`](dch_data/dvs_gesture.py)

- Simple event-to-spike encoder (torch-optional fallback)  
  [`SimpleBinnerEncoder`](dch_data/encoders.py:62)

- Dataset downloader CLI (lazy tonic import; JSON output)  
  [`scripts/download_datasets.py`](scripts/download_datasets.py)

---

## Backend Abstraction Layer (SNN)

Torch is optional. Interfaces remain import-safe and usable without torch.

- Encoder/Model/Trainer configs and protocols  
  [`EncoderConfig`](dch_snn/interface.py:51), [`ModelConfig`](dch_snn/interface.py:64), [`TrainerConfig`](dch_snn/interface.py:76),  
  [`Encoder`](dch_snn/interface.py:97), [`SNNModel`](dch_snn/interface.py:131), [`Trainer`](dch_snn/interface.py:178), [`BackendAdapter`](dch_snn/interface.py:216)

---

## Configuration

Plain YAML (no runtime Hydra dependency) for experiments, DCH params, scaffolding, FSM, and sweeps:

- Experiments: `configs/experiments/dvs_gesture.yaml`, `configs/experiments/nmnist.yaml`  
- DCH & Pipeline defaults: `configs/dch.yaml`, `configs/pipeline.yaml`  
- Scaffolding: `configs/scaffolding.yaml`  
- FSM: `configs/fsm.yaml`  
- Sweeps: `configs/hyperparams_sweep.yaml`  
- CV splits: `configs/cv.yaml`, generator [`scripts/make_splits.py`](scripts/make_splits.py)

---

## Benchmarks (Stdlib-only, deterministic)

- Traversal micro-benchmark → JSON line  
  [`benchmarks/benchmark_traversal.py`](benchmarks/benchmark_traversal.py)

- Pipeline macro-benchmark → JSON line  
  [`benchmarks/benchmark_pipeline.py`](benchmarks/benchmark_pipeline.py)

---

## Testing

Fast, torch-free unit tests cover DHG, traversal, plasticity, FSM, embeddings, transforms, gating, logging, and integration smoke.

- Start with:  
  `pytest -q`  
  Selected: `pytest -q tests/test_pipeline_smoke.py` (torch-free path)

- Integration (torch-required) are skipped automatically when torch is absent.

---

## Design Notes

- Determinism-first: seeded RNGs, reproducible traversal, and logging in machine-readable formats.
- Optional dependencies are lazily imported with actionable error messages; modules import cleanly without them.
- The pipeline can be feature-gated (e.g., `enable_abstraction`) to bound compute and keep default behavior unchanged.

---

## Extending DCH

1) Implement a protocol from [`dch_core/interfaces.py`](dch_core/interfaces.py) (e.g., a custom `TraversalEngine` or `EmbeddingEngine`).
2) Wire it into the pipeline via composition (no global state).
3) Add unit tests (torch-free when possible); mock optional deps.
4) Update `docs/CONTRIBUTING.md` and `docs/API_REFERENCE.md` with new public APIs.
