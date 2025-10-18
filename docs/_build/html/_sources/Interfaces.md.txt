# Interface Contracts and Typed Data Models for DCH

Status: Draft v0.1
Date: 2025-10-04
Owners: DCH Maintainers

Summary
- Defines typed data models and contracts for events, hypergraph structures, plasticity states, pipelines, and backend adapters.
- Establishes Backend Abstraction Layer (BAL) for Norse primary and BindsNET parity.
- Maps responsibilities to files for subsequent implementation.

Type aliases and identifiers
- VertexId: 64-bit integer or canonical string; must be unique within a run.
- EdgeId: 64-bit integer or canonical string; unique within a run.
- NeuronId: integer in [0, N).
- Timestamp: int64 in microseconds (dataset native time unit mapped atomically).
- ReliabilityScore: float in [0.0, 1.0].
- Window: closed interval [t0, t1] with t0 <= t1.

Entity: Event
- Fields: neuron_id (NeuronId), t (Timestamp), meta (dict, optional).
- Invariants: t is non-decreasing within a source stream; meta keys are JSON-serializable.
- Serialization: JSON record with keys neuron_id, t, meta.
- Failure modes: out-of-order timestamps must be handled via buffering or dropped per policy.

Entity: Vertex (spike event materialized)
- Fields: id (VertexId), neuron_id (NeuronId), t (Timestamp).
- Invariants: id bijective with (neuron_id, t) per stream; t monotone across id ordering.
- Indexing: time-ordered storage and per-neuron ring buffers.

Entity: Hyperedge (causal hypothesis)
- Fields: id (EdgeId), tail (set[VertexId] non-empty), head (VertexId), delta_min (int), delta_max (int), refractory_rho (int), reliability (ReliabilityScore), counts_success (int), counts_miss (int), last_update_t (Timestamp).
- Invariants: head not in tail; delta_min <= delta_max; refractory observed for same head neuron.
- Semantics: traversable if all tail vertices are present (B-connectivity) and temporal constraints satisfied.
- Initialization: reliability near prior (e.g., 0.1), counts initialized to 0.
- Pruning: remove when reliability < threshold for K consecutive reviews or by budget policy.

Entity: Hypergraph
- Fields: V (map[VertexId -> Vertex]), E (map[EdgeId -> Hyperedge]), incoming (map[VertexId -> set[EdgeId]]), outgoing (map[VertexId -> set[EdgeId]]), time_index (ordered structure for window queries).
- Operations (contracts):
  - ingest_event: add Vertex and update indices; return VertexId.
  - generate_candidates_tc_knn: return candidate Hyperedges for a new head.
  - insert_hyperedges: admit candidates with dedup and budgets.
  - window_query: fetch vertices/hyperedges in Window.
  - prune: drop edges below thresholds and rebalance indices.
- Complexity targets: amortized near O(log M) for inserts into time_index; candidate generation budgeted by k and window size.

Entity: Hyperpath
- Representation: DAG-like object capturing a head and its validated antecedents via a set of Hyperedges.
- Score: product or minimum of member reliability scores with optional length penalty.
- Canonical label: stable string used by FSM for frequent pattern counting.

Entity: PlasticityState
- Fields: ema_alpha (float), reliability_clamp (tuple[min, max]), decay_lambda (float), freeze (bool), prune_threshold (float).
- Invariants: reliability within clamp; freeze disables updates for protected edges.

Backend Abstraction Layer (BAL) contracts
- Model IO: backend must consume spike tensors/streams and emit layer spikes and task outputs.
- Encoder: converts Events into backend-ready tensors with consistent timing semantics.
- Trainer: performs task-dependent updates without interfering with DCH credit assignment.
- Registry: resolve backend key to adapter and validate availability.
- Checkpointing: uniform metadata schema for cross-backend restore.

Pipeline stage contracts (inputs, outputs, invariants)
- Preprocessing: raw dataset -> normalized events; invariant: timestamps preserved.
- Encoding: events -> spike tensors; invariant: shape and time base consistent with backend.
- DHG construction: spikes + connectivity -> candidate hyperedges; invariant: temporal window and refractory enforced.
- Traversal: target vertex -> set of valid hyperpaths; invariant: B-connectivity and temporal logic satisfied.
- Credit assignment: hyperpaths -> edge evidence; invariant: discrete, evidence-based updates.
- Plasticity: evidence -> reliability updates and pruning; invariant: clamps and thresholds respected.
- Embedding: hyperpath -> embedding vector; invariant: stable under isomorphic reorderings.
- FSM: embeddings/labels -> frequent pattern promotions; invariant: sliding window and hysteresis applied.
- Abstraction: frequent chain -> higher-order edge; invariant: acyclic and non-duplicative.
- Scaffolding: task similarity -> freeze/grow policy; invariant: provenance recorded.
- Evaluation: metrics and stats with seeds logged.

Error handling and logging
- All stages must surface structured errors with context (dataset, window, head vertex id).
- Logging includes seeds, config fingerprints, counts of generated/pruned edges, and traversal statistics.

Determinism and seeds
- Single source of truth for seeds; disable nondeterministic kernels when configured.
- Record environment fingerprint and library versions with each run.

Serialization contracts
- Events: line-delimited JSON or Parquet for batch export.
- Hypergraph: snapshot as JSON with vertices, hyperedges, indices; large runs may stream deltas.
- Artifacts: metrics CSV, TensorBoard logs, config YAML, environment JSON, checkpoints.

Concurrency and memory
- Ingest and traversal operate on lock-free or coarse-grain locked indices with bounded latency.
- Packed Memory Array or similar structure recommended for dynamic indices.

File mapping for implementation
- [dch_core/interfaces.py](dch_core/interfaces.py)
- [dch_snn/interface.py](dch_snn/interface.py)
- [dch_snn/registry.py](dch_snn/registry.py)
- [dch_pipeline/pipeline.py](dch_pipeline/pipeline.py)
- [dch_pipeline/evaluation.py](dch_pipeline/evaluation.py)
- [dch_pipeline/stats.py](dch_pipeline/stats.py)
- [dch_pipeline/seeding.py](dch_pipeline/seeding.py)
- [dch_data/encoders.py](dch_data/encoders.py)
- [dch_core/dhg.py](dch_core/dhg.py)
- [dch_core/traversal.py](dch_core/traversal.py)
- [dch_core/plasticity.py](dch_core/plasticity.py)
- [dch_core/embeddings/wl.py](dch_core/embeddings/wl.py)
- [dch_core/fsm.py](dch_core/fsm.py)
- [dch_core/abstraction.py](dch_core/abstraction.py)
- [dch_core/scaffolding.py](dch_core/scaffolding.py)

Acceptance criteria for interface stability
- Contracts implemented in the above files with unit tests covering success and failure cases.
- Backends register via registry and pass BAL contract tests on CPU.
- Serialization round-trips verified for events and hypergraph snapshots.
- Determinism validated across seeds on small subsets.

Change control
- Breaking changes require version bump and migration notes.
- Contracts documented in API reference and traced to module responsibility matrix.

References
- See [docs/FrameworkDecision.md](docs/FrameworkDecision.md) and [docs/PaperPackaging.md](docs/PaperPackaging.md) for related policies.

End of spec