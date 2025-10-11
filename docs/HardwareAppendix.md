# Hardware Mapping Appendix — “Causa‑Chip” Sketch

Purpose: summarize how DCH’s core primitives map to a plausible neuromorphic “Causa‑Chip” comprised of lightweight PIM and streaming units. This appendix is informational; it does not introduce runtime requirements.

---

## Components and computational primitives

- GSE — Graph Structure Extraction
  - Software analog: candidate construction in [dch_core/dhg.py](dch_core/dhg.py) and windowed neighbor queries in the hypergraph store.
  - Function: TC‑kNN over recent presyn spikes, temporal admissibility checks, tail enumeration under causal window δ.
  - Primitives: windowed index lookup, top‑k selection, set/dedup operations, timestamp comparisons.
  - Possible hardware: small CAM/PMA index for recent events, SRAM ring buffers, priority queues for top‑k.

- GMF — Graph Memory Fabric
  - Software analog: in‑memory store, adjacency, and pruning in [dch_core/hypergraph_mem.py](dch_core/hypergraph_mem.py).
  - Function: append‑only vertex ingestion, hyperedge admission, reliability field storage, snapshots.
  - Primitives: pointer‑chasing adjacency, scatter/gather updates, thresholded pruning.
  - Possible hardware: banked SRAM for vertices/edges, compressed adjacency lists, tag bits for active/pruned edges.

- PTA — Plasticity and Temporal Aggregation
  - Software analog: reliability updates in [dch_core/plasticity.py](dch_core/plasticity.py).
  - Function: evidence aggregation over hyperpaths, EMA updates with clamping, counters, decay.
  - Primitives: vector EMA (a*x + b*y), clamp, sparse scatter, periodic decay sweep.
  - Possible hardware: SIMD MAC lanes over reliability arrays; low‑precision fixed‑point viable.

- FSM Engine — Streaming Frequent Hyperpath Mining
  - Software analog: decayed counting and promotion in [dch_core/fsm.py](dch_core/fsm.py).
  - Function: decay+hysteresis counters for hyperpath labels, thresholded promotions to “frequent”.
  - Primitives: per‑label decayed counters, compare‑and‑promote, small priority structures.
  - Possible hardware: on‑chip counters with background decay, small LUTs for labels.

- Meta‑Controller
  - Software analog: orchestration/looping in [dch_pipeline/pipeline.py](dch_pipeline/pipeline.py), policies in [dch_core/scaffolding.py](dch_core/scaffolding.py), abstraction hooks in [dch_core/abstraction.py](dch_core/abstraction.py).
  - Function: step scheduling (ingest → DHG → traversal → plasticity → FSM → optional abstraction), rate limiting, checkpoint/snapshot control.
  - Primitives: event‑driven FSM, timers, back‑pressure signals, configuration registers.

---

## Data structures and PIM alignment

- Event ring buffers
  - Time‑sorted buffers for recent spikes (presyn/postsyn), with head/tail pointers per neuron.
  - Operations: window queries, refractory guards.

- Hyperedge adjacency
  - Compressed tail lists per head with reliability, timestamps, and labels.
  - Operations: tail expansion, deduplication, pruning.

- PMA (Parallel Memory Array) indexing
  - Small associative structures to accelerate TC‑kNN and window membership tests.
  - Ops: range queries by timestamp, sparse set intersections.

- ReRAM / SRAM PIM kernels
  - SpMV: reliability‑weighted message aggregation along edges for credit signals.
  - SpGEMM: combining tail‑tail interactions for higher‑order tail enumeration and label composition.
  - Notes: dynamic sparsity implies indirection; a hybrid “gather‑compute‑scatter” is often more practical than pure crossbar.

---

## Mapping DCH operations to kernels

- DHG candidate generation (GSE + GMF)
  - Sliding‑window neighbor lookup → indexed gather
  - Tail enumeration and dedup → set ops + bounded combinations
  - Admission with budgets → counters + thresholding

- Backward traversal and credit (Traversal)
  - Beam search with B‑connectivity → iterative sparse gather + top‑k select
  - Score composition with reliability → vector MAC and min/agg combiners

- Plasticity (PTA)
  - EMA updates → elementwise fused multiply‑add + clamp
  - Decay and prune → background sweep + thresholded write‑back

- FSM promotion
  - Decayed label counters → periodic multiply‑by‑λ and compare‑with‑θ
  - Hysteresis → dual thresholds or hold‑k queues

- Abstraction
  - Promotion of frequent hyperpaths to higher‑order edges → batched writes to adjacency with dedup guards

---

## Feasibility and research gaps

- Dynamic sparse structure updates
  - Challenge: on‑chip CSR/adjacency maintenance under frequent inserts/prunes.
  - Direction: over‑provisioned banks with periodic compaction; commit logs with deferred merge.

- Labeling and canonicalization
  - Need low‑cost canonical labeling for hyperpaths (used in traversal/FSM).
  - Direction: hash‑based labels with collision monitoring rather than full canonical forms.

- Memory budgets and precision
  - Fixed‑point is likely sufficient for reliability/EMA counters.
  - Evaluate precision/overflow bounds for decay and hysteresis to preserve stability.

- Scheduling and QoS
  - Maintain predictable latency under bursty event streams.
  - Use beam/combination budgets (already in software) as hardware rate limiters.

---

References to software components:
- DHG: [dch_core/dhg.py](dch_core/dhg.py)
- Traversal: [dch_core/traversal.py](dch_core/traversal.py)
- Plasticity: [dch_core/plasticity.py](dch_core/plasticity.py)
- FSM: [dch_core/fsm.py](dch_core/fsm.py)
- Abstraction: [dch_core/abstraction.py](dch_core/abstraction.py)
- Scaffolding: [dch_core/scaffolding.py](dch_core/scaffolding.py)
- Orchestration: [dch_pipeline/pipeline.py](dch_pipeline/pipeline.py)