# Dynamic Causal Hypergraph DCH — Section 10 Complexity and Resource Model

Parent outline [DCH_TechSpec_Outline.md](../DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](../sections/DCH_Section1_FormalFoundations.md), Section 2 [DCH_Section2_DHG_TCkNN.md](../sections/DCH_Section2_DHG_TCkNN.md), Section 3 [DCH_Section3_Plasticity.md](../sections/DCH_Section3_Plasticity.md), Section 4 [DCH_Section4_HyperpathEmbedding.md](../sections/DCH_Section4_HyperpathEmbedding.md), Section 5 [DCH_Section5_CreditAssignment.md](../sections/DCH_Section5_CreditAssignment.md), Section 6 [DCH_Section6_FSM.md](../sections/DCH_Section6_FSM.md), Section 7 [DCH_Section7_HierarchicalAbstraction.md](../sections/DCH_Section7_HierarchicalAbstraction.md), Section 8 [DCH_Section8_TaskAwareScaffolding.md](../sections/DCH_Section8_TaskAwareScaffolding.md), Section 9 [DCH_Section9_Interfaces.md](../sections/DCH_Section9_Interfaces.md)

Version v0.1

1. Purpose and scope  
- Provide time and space complexity for each DCH subsystem with practical throughput targets on event vision workloads DVS Gesture and N-MNIST.  
- Dimension memory and latency budgets; define backpressure and adaptive control to keep the system within real-time constraints.

2. Workload characterization and symbols  
Workloads  
- N-MNIST: events per second λ ≈ 1e4–1e5; bursts up to 5e5/s for tens of ms.  
- DVS Gesture: λ ≈ 1e5–1e6; bursts up to 2e6/s for tens of ms.  
Symbols  
- N: number of neurons; typically 1e3–1e5 in prototypes.  
- E: active hyperedges |E(t)|.  
- d_in: average admissible in-degree after temporal filters (Section 2).  
- b: average ring-buffer items in window per neuron.  
- k_max: max tail size (default 3).  
- F_max: WL frontier cap (default 256).  
- K: seeds/beam width (default 8).  
- L: traversal depth cap (default 12).  
- C_in: per-vertex admissible cap (default 16).  
- W: FSM window (default 60 s).  
- γ: global decay (default 0.98/s).

3. Per-module time complexity and latency targets  
3.1 DHG TC-kNN (Section 2)  
- Per post-spike: O(d_in log b + c), where c ≤ combinations admitted ≤ M_in + C_cap (defaults 6 and 10).  
- Latency target desktop: ≤ 100 μs/event; embedded: ≤ 200 μs/event.  
- Determinants: Pred j degree, window [Δ_min, Δ_max], δ_causal, dedup hash cost.

3.2 Plasticity (Section 3)  
- Watcher creation: O(|Tail|). Resolution: O(1) updates to discounted counters and EMA.  
- Housekeeping decay/prune: amortized O(#edges updated per tick).  
- Latency target: ≤ 50 μs/event for watcher ops; prune tick ≤ 2 ms per second.

3.3 WL embedding and grouping (Section 4)  
- Update per event: O(|frontier|) ≤ O(F_max) + O(bands) for LSH inserts.  
- Latency target: ≤ 200 μs/event; bounded by F_max and hashing constants.  
- SAGE refresh (periodic): 500 ms cadence; ≤ 20 ms per refresh on GPU.

3.4 Traversal and credit assignment (Section 5)  
- Per seed: O(L · C_in) with softmax scoring; with beam K: O(K · L · C_in).  
- Latency target: ≤ 1 ms/seed; with 8 seeds ≤ 8 ms per traversal cycle.  
- Cycle cadence: every T_trav (config default 20 ms) or on supervision/reward.

3.5 FSM canonicalize+count (Section 6)  
- Per path: O(|π| · avg_tail) canonicalization with small constants + O(1) HH/CMS updates.  
- Latency target: ≤ 50 μs/path; maintenance tick ≤ 10 ms/s.

3.6 Abstraction HOEs (Section 7)  
- DHG-time matching: O(m log b) for m=|Tail_HO| (≤ k_max_rule).  
- Traversal compression: replaces multiple expansions with O(1).  
- Latency target: ≤ 30 μs/HOE match attempt.

3.7 Scaffolding (Section 8)  
- Calibration similarity: O(|B_task| + |P|) using set ops and cosine on d=64 (WL).  
- Policy step: O(#knobs) negligible at 100–500 ms cadence.

4. End-to-end latency budget per event (desktop prototype)  
- DHG 100 μs  
- Plasticity 50 μs  
- WL update 200 μs  
- Sum ≤ 350 μs per event on average path.  
- Traversal, FSM, and abstraction run on separate cadences; ensure their amortized compute does not stall event path (use separate threads/queues).

5. Throughput targets and headroom  
- N-MNIST: ≥ 1e5 events/s sustained with ≥ 2× headroom (≥ 2e5/s).  
- DVS Gesture: ≥ 5e5 events/s sustained with ≥ 1.5× headroom (≥ 7.5e5/s).  
- Bursts: tolerate 2× burst for ≥ 50 ms via buffering without drop; backpressure engages beyond that.

6. Memory model and sizing  
6.1 Per-object footprints (approximate)  
- Vertex record: 16 B (id, neuron id, timestamp).  
- Event-anchored hyperedge: 96–128 B (ids, timings, reliability, counters, pointers).  
- Index entries: 16–24 B per edge per index (incoming-by-head, tail-signature).  
- WL vector per vertex: d=64 floats quantized to 16-bit (128 B) + metadata 16 B ≈ 144 B.  
- SAGE snapshot per vertex: d=128 float16 ≈ 256 B (periodic in memory/disk).  
- HH template: 24 B stats + 16 B id + HLL 16 B ≈ 56 B; CMS: width × depth counters (4 B each).

6.2 Working set budgets  
Desktop prototype targets  
- |E| ≤ 50 M edges total cap; practical working set 5–20 M edges.  
- Memory estimate (E=10 M):  
  - Edge core 10 M × 112 B ≈ 1.12 GB  
  - Indexes 2 × 10 M × 20 B ≈ 0.40 GB  
  - WL embeddings for K_active vertices (e.g., 5 M) 5 M × 144 B ≈ 0.72 GB  
  - FSM HH (100k) ≈ 5.6 MB; CMS (32768×4×4 B) ≈ 2.1 MB  
  - Total ≈ 2.3–2.5 GB plus buffers and overhead ⇒ fits in 32 GB comfortably; scale to E=50 M ≈ 11–13 GB.  
Embedded targets  
- |E| ≤ 5 M; total ≤ 2–3 GB including embeddings and indices on 8–16 GB systems (reduce d, HH size, and snapshot retention).

6.3 Buffering and queues  
- Per-neuron ring buffers: choose T_ret so b ≈ λ · (Δ_max) per neuron; bound per-neuron items N_ret to cap memory (e.g., 64–256).  
- Event queues sized for 100 ms at target throughput; desktop: ≥ 1e5 events × 0.1 s = 1e4 slots; use lock-free MPMC.

7. Backpressure and adaptive control  
- Admission throttling: raise τ_prune and lower M_in, C_cap when queue lag > L_q_high.  
- Traversal duty-cycling: increase T_trav; reduce K and L under load.  
- FSM throttling: raise s_min and s_path_min during churn; decay faster (γ↑).  
- WL cadence: increase Δt_WL temporarily; cap F_max.  
- Meta-controller policies publish these knob changes at 10–50 ms cadence.

8. Scaling laws and sensitivity  
- DHG cost scales ~ O(λ · d_in log b) per event; cap d_in with tighter windows and regionization.  
- Traversal cost scales ~ O(rate_seeds · K · L · C_in); seeds tied to supervision frequency, not per-event.  
- FSM cost scales with path emission rate; keep |π| small via HOE compression.  
- Memory scales linearly in |E| and #active vertices; manage with pruning and HOEs.

9. Capacity planning for benchmarks  
N-MNIST (λ=1e5/s nominal)  
- Event path budget 350 μs ⇒ CPU usage ~ 35%/core if single-thread; parallelize across 8 cores to ample headroom.  
- Traversal every 20 ms with 8 seeds and ~500 paths per second; FSM < 25k paths/s capability.  
DVS Gesture (λ=5e5/s nominal)  
- Event path budget requires parallel DHG/WL lanes; 4–8 worker shards by neuron id hashing.  
- Traversal batch sizes reduced under load (K=6, L=10) and cadence T_trav=30 ms; FSM s_min=75 to control pattern rate.

10. Performance diagram and ownership

```mermaid
flowchart LR
EVQ[Event queue] --> DHG[DHG 100us]
DHG --> PL[Plasticity 50us]
PL --> WL[WL update 200us]
WL --> OUT[End of per-event path]
OUT --> TRAV[Traversal cycle 8ms]
TRAV --> FSM[FSM tick 10ms/s]
FSM --> ABS[Abstraction match 30us]
```

11. Acceptance criteria for Section 10  
- Complexity expressions per module are provided and align with earlier sections.  
- Concrete latency and throughput targets set for N-MNIST and DVS Gesture.  
- Memory budgets dimensioned for desktop and embedded with numeric examples.  
- Backpressure and adaptive policies specified.  
- Diagram reflects performance pipeline and budgets.

12. Cross references  
- DHG costs and parameters [DCH_Section2_DHG_TCkNN.md](../sections/DCH_Section2_DHG_TCkNN.md)  
- Plasticity and pruning [DCH_Section3_Plasticity.md](../sections/DCH_Section3_Plasticity.md)  
- Embeddings [DCH_Section4_HyperpathEmbedding.md](../sections/DCH_Section4_HyperpathEmbedding.md)  
- Traversal [DCH_Section5_CreditAssignment.md](../sections/DCH_Section5_CreditAssignment.md)  
- FSM [DCH_Section6_FSM.md](../sections/DCH_Section6_FSM.md)  
- Abstraction [DCH_Section7_HierarchicalAbstraction.md](../sections/DCH_Section7_HierarchicalAbstraction.md)  
- Scaffolding [DCH_Section8_TaskAwareScaffolding.md](../sections/DCH_Section8_TaskAwareScaffolding.md)  
- Interfaces for metrics and knobs [DCH_Section9_Interfaces.md](../sections/DCH_Section9_Interfaces.md)

End of Section 10