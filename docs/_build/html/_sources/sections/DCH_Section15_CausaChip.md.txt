# Dynamic Causal Hypergraph DCH — Section 15 Causa Chip Hardware Co Design Overview

Parent outline [DCH_TechSpec_Outline.md](../DCH_TechSpec_Outline.md)  
Cross references [DCH_Section10_ComplexityResource.md](../sections/DCH_Section10_ComplexityResource.md), [DCH_Section2_DHG_TCkNN.md](../sections/DCH_Section2_DHG_TCkNN.md), [DCH_Section5_CreditAssignment.md](../sections/DCH_Section5_CreditAssignment.md), [DCH_Section6_FSM.md](../sections/DCH_Section6_FSM.md), [DCH_Section7_HierarchicalAbstraction.md](../sections/DCH_Section7_HierarchicalAbstraction.md), [DCH_Section9_Interfaces.md](../sections/DCH_Section9_Interfaces.md)

Version v0.1

1. Objectives and scope
- Translate DCH computational primitives into a heterogeneous SoC architecture for low latency event stream processing.
- Provide first order bandwidth latency and capacity estimates for DVS Gesture and N MNIST targets.
- Define unit responsibilities, interfaces, NoC interconnect, memory hierarchy, and fallback software model.

2. Workload recap and design targets
- Event lane latency budget per event ≤ 350 microseconds desktop prototype see [DCH_Section10_ComplexityResource.md](../sections/DCH_Section10_ComplexityResource.md).
- Throughput sustained  
  - N MNIST ≥ 1e5 events per second with 2x headroom.  
  - DVS Gesture ≥ 5e5 events per second with 1.5x headroom.
- Traversal cycle ≤ 8 ms per batch of seeds; FSM tick ≤ 10 ms per second maintenance.
- Memory working set edges 5 to 20 M typical desktop profile, up to 50 M cap see Section 10.

3. Architectural overview
- Heterogeneous SoC with five primary accelerators and one control core on a high bandwidth NoC:
  - Graph Streaming Engine GSE for event ingestion and TC kNN candidate generation.
  - Graph Memory Fabric GMF a PIM like memory subsystem for dynamic hypergraph storage and atomic updates.
  - Parallel Traversal Accelerator PTA for constrained backward hyperpath traversal and credit aggregation.
  - Frequent Subgraph Miner FSM engine for canonical labeling counting and streaming promotion.
  - Meta Controller MC low power control processor to coordinate knobs, backpressure, and policy.
- Shared chip level SRAM and off chip DRAM interface; optional on package HBM for high end.

4. Computational unit specifications

4.1 Graph Streaming Engine GSE
- Function  
  - Ingest events, maintain per neuron ring buffers, perform presyn lookup, temporal window search, candidate scoring, deduplication, refractory checks, and insert requests to GMF.
- Microarchitecture  
  - Hardware event queues with timestamp watermarking, per neuron ring buffers in SRAM with binary search assist.  
  - TC kNN pipeline stages fetch presyn indices, window filter, combine tails up to k max 3, score and select with budgets M in and C cap.  
  - Signature unit computes tail signature hashes.
- Interfaces  
  - Read only access to static synapse map Pred j, read write to GMF edge tables via GMF message ports.  
  - Metrics counters for candidate hit rate dedup and latency.

4.2 Graph Memory Fabric GMF
- Function  
  - Primary storage for event anchored edges, template HOEs, and indices incoming by head and by tail signature.  
  - Provide in memory atomic operations on reliability and counters; support dynamic insert delete and prune.
- Microarchitecture  
  - PIM near memory arrays e.g., ReRAM like banks plus digital periphery for atomic add min max and bounded clip on per edge fields.  
  - Packed Memory Array PMA like data structure controller for batched inserts and compaction.  
  - Two level index caches small SRAM for hot indices and larger eDRAM or SRAM for mid tier.
- Operations  
  - Insert edge tail list and head id, write attributes, update two indices and creation token.  
  - Atomic EMA update r e and counters with bound checks.  
  - Prune by priority queue scan on s prune equals r e times freshness decays.

4.3 Parallel Traversal Accelerator PTA
- Function  
  - Execute multi start randomized beam style backward traversal with B connectivity and temporal logic constraints; accumulate edge contributions A e.  
- Microarchitecture  
  - PE array arranged as 2D mesh each PE handles a partition of head vertex space; local caches for incoming by head lists and edge attributes; frontier and beam queues per PE.  
  - Edge filter units check Δ windows refractory and horizon; scorer computes softmax features w rel w rec w rule w sim.
- Data path  
  - Pull candidate edges from GMF via read bursts; write contributions to per edge accumulators in GMF or scratchpad then reduce.  
  - Deterministic RNG per PE seeded by seed id and snapshot cycle.
- Controls  
  - Caps L depth B branching C in admissible per vertex, K beams; early termination based on upper bound estimate.

4.4 Frequent Subgraph Miner FSM engine
- Function  
  - Canonicalize grounded hyperpaths to template ids, maintain heavy hitters SpaceSaving and CMS with sliding window and decay, detect concept drift, and emit promotions.
- Microarchitecture  
  - Canonical labeling pipeline encode lag bucketization, incidence expansion layering, token serialization; 128 bit hash unit with collision chains in on chip CAM.  
  - Counting stage with multi bank CAM and SRAM hash tables for HH; CMS arrays in SRAM with decay logic.  
  - HyperLogLog sketchers for coverage per HH template.
- Output  
  - Rule descriptors to MC, HOE construction triggers to GMF via Abstraction microservice.

4.5 Meta Controller MC
- Function  
  - Execute policy loops for backpressure and scaffolding knobs; orchestrate cadence and budgets; log and expose health.  
- Microarchitecture  
  - RISC V class core or small microcontroller, access to module counters and control registers over NoC; interrupt lines for threshold breaches.

5. Network on Chip NoC and memory hierarchy
- NoC  
  - 2D mesh or ring bus hybrid; express routes between GSE GMF and PTA; QoS channels high priority for event lane traffic.  
  - Bandwidth target aggregate ≥ 200 GB per second desktop class; arbitration priority to event path reads writes.
- Memory  
  - On chip SRAM 8 to 64 MB for indices caches queues and scratch; optional eDRAM blocks for mid tier.  
  - Off chip DDR5 or LPDDR5X 16 to 32 GB system memory; for high end consider HBM3 256 to 512 GB per second.
- Atomicity  
  - GMF provides line level atomic operations for reliability and counters; PIM reduces round trips on hot updates.

6. Bandwidth and latency model first order
- GSE  
  - Reads presyn indices O d in and ring buffer heads; writes new edges and index updates; budget ≤ 100 microseconds per event.  
  - Estimated traffic per admitted edge insert metadata 64 to 128 B plus two index writes 16 to 32 B each.
- PTA  
  - For K beams L depth and C in caps worst case touches K times L times C in edge records; with caching and admissibility filters effective fetches smaller.  
  - Target ≤ 1 ms per seed; aggregate ≤ 8 ms cycle; batched read bursts amortize header overhead.
- FSM  
  - Per path canonicalization tens of tokens 1 to 2 cache lines; HH and CMS O 1 updates; maintenance tick linear in HH size 100k.  
  - Budget ≤ 50 microseconds per path hot path; maintenance ≤ 10 ms per second.
- NoC  
  - Event path peak concurrency dimensioned to tolerate bursts 2x typical for ≥ 50 ms with buffering and priority QoS.

7. Capacity and sizing examples rough order of magnitude at 7 nm for guidance only
- Area budgets indicative  
  - GSE 8 to 15 mm^2 ring buffers, signature and TC kNN pipeline.  
  - GMF per bank 10 to 20 mm^2 excluding large SRAM eDRAM; replicated 4 to 8 banks.  
  - PTA PE array 20 to 40 mm^2 including local caches.  
  - FSM pipeline 8 to 12 mm^2 including CAM and counters.  
  - MC and glue logic 2 to 4 mm^2.  
  - Total compute logic 60 to 100 mm^2 excluding large memory macros.
- Power envelopes indicative at 1 GHz class clocks  
  - Event lane sustained 10 to 25 W depending on |E| and λ; PTA spikes to 5 to 10 W during traversal cycles.  
  - FSM 1 to 3 W average; MC negligible.  
  - Aggressive power gating for PTA and FSM between bursts.
- Latency  
  - End to end event path ≤ 350 microseconds average; p99 depends on NoC contention; enforce QoS.

8. Programming and runtime model
- Driver model  
  - Modules exposed via memory mapped control registers and DMA queues; host driver schedules cadences and monitors counters.  
  - Snapshot ids for WL SAGE and FSM rules tracked in control plane per [DCH_Section9_Interfaces.md](../sections/DCH_Section9_Interfaces.md).
- Data flow  
  - Event DMA to GSE, GMF handles inserts; PTA launched with seeds; FSM consumes path stream; MC adjusts knobs on thresholds.
- Fallback  
  - Software fallback for PTA and FSM on host CPU GPU during bringup; deterministic results matched via snapshot ids.

9. Abstraction and HOE handling in hardware
- HOE table in GMF with role definitions lag intervals and reliability; small CAM for fast role matching.  
- GSE consults HOE table to emit HOE candidates with boosted scores.  
- PTA treats HOE as single step edges retrieving provenance pointers only if audit requested.

10. Flow and data path diagrams

10.1 Chip level dataflow

```mermaid
flowchart LR
EVQ[Event Queues] --> GSE[GSE TC-kNN]
GSE --> GMF[GMF Graph Memory Fabric]
GMF --> PTA[PTA Traversal]
PTA --> FSME[FSM Engine]
FSME --> MC[Meta Controller]
MC -->|knobs| GSE
MC -->|knobs| PTA
MC -->|knobs| FSME
MC -->|prune| GMF
```

10.2 GMF update pipeline

```mermaid
flowchart TB
REQ[Insert or Update Request] --> IDX[Index Lookup]
IDX --> PIMU[PIM Atomic Update r_e and Counters]
PIMU --> IDX2[Update Incoming-by-Head and Tail-Signature]
IDX2 --> ACK[Ack and Metrics]
```

10.3 PTA PE array topology

```mermaid
flowchart TB
SEEDS[Seed Queue] --> DISPATCH[Dispatcher]
DISPATCH --> PE1[PE Partition 1]
DISPATCH --> PE2[PE Partition 2]
DISPATCH --> PEn[PE Partition N]
PE1 --> REDUCE[Reduce Edge Contributions]
PE2 --> REDUCE
PEn --> REDUCE
REDUCE --> GMFWR[Write Aggregates to GMF]
```

10.4 FSM pipeline

```mermaid
flowchart LR
PATHS[Grounded Hyperpaths] --> NORM[Normalize and Lag Bucketize]
NORM --> CANON[Canonical Labeling + Hash]
CANON --> COUNT[HH + CMS Update]
COUNT --> THRES[Threshold + Hysteresis]
THRES --> RULES[Promote/Demote Rules]
```

11. Interfaces mapping to software spec
- Each hardware module implements a subset of the APIs in [DCH_Section9_Interfaces.md](../sections/DCH_Section9_Interfaces.md)  
  - GSE maps to dhg.on_post_spike, get params, metrics snapshot.  
  - GMF maps to hyperedge CRUD and atomic updates.  
  - PTA maps to traversal.assign_credit with hardware beam and caps.  
  - FSM maps to fsm.submit_path and fsm.tick and poll promotions.  
  - MC maps to meta.step and scaffolding style knob pushes.

12. Backpressure and fault handling on chip
- Hardware thresholds for queue depth watermark lag and memory occupancy drive MC actions: lower C cap raise τ prune and stretch Δt WL cadences.  
- Fault containment  
  - If GMF rejects inserts due to capacity pressure MC pauses GSE candidate combinations first; then raises pruning.  
  - If PTA exceeds latency budget MC reduces K L and C in and raises rule bias; can also skip noncritical seeds.

13. Verification and co simulation strategy
- Trace driven simulation using software DCH to generate event streams and golden paths.  
- Unit level verification testbenches for GSE GMF PTA FSM with corner cases Δ windows refractory checks and collision storms.  
- NoC contention and roofline models to validate bandwidth headroom.  
- Determinism tests seeded RNG and snapshot checksums.

14. Risks and open questions hardware
- PIM primitive fidelity for bounded clip EMA and decay operations verify numerical stability and endurance.  
- FSM canonicalization throughput ensure pipeline depth and CAM sizing to sustain target path rates without backpressure.  
- Partitioning strategy for PTA to minimize remote edge fetches consider vertex partitioning by head id and time sharding.  
- HOE matchers complexity vs benefit ensure CAM size sufficient for active HOE set K HOE global.

15. Early BoM style estimates non binding
- Desktop accelerator card  
  - Die 120 to 180 mm^2 compute logic plus memory macros; TDP 25 to 50 W depending on configuration.  
  - 16 to 32 GB LPDDR5X off chip; optional HBM variant for research.  
- Embedded edge module  
  - Die 60 to 90 mm^2 with reduced GMF and PTA; TDP 5 to 12 W; 8 to 16 GB LPDDR5X.

16. Acceptance criteria for Section 15
- Units defined with roles microarchitecture operations and interfaces.  
- NoC and memory hierarchy sketched with bandwidth targets.  
- First order area power and latency budgets provided.  
- Dataflow and pipeline diagrams included.  
- Fault and backpressure policies mapped to MC actions.

17. Cross references
- Complexity and performance budgets [DCH_Section10_ComplexityResource.md](../sections/DCH_Section10_ComplexityResource.md)  
- Abstraction and HOE usage [DCH_Section7_HierarchicalAbstraction.md](../sections/DCH_Section7_HierarchicalAbstraction.md)  
- FSM requirements [DCH_Section6_FSM.md](../sections/DCH_Section6_FSM.md)  
- Traversal constraints [DCH_Section5_CreditAssignment.md](../sections/DCH_Section5_CreditAssignment.md)  
- Interface contracts [DCH_Section9_Interfaces.md](../sections/DCH_Section9_Interfaces.md)

End of Section 15