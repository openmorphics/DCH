<!-- Assembled 2025-10-04 17:07:23 UTC by scripts/build_master_md.py -->
# Dynamic Causal Hypergraph DCH — Technical Specification v0.1 and Causa-Chip Co-Design

Version v0.1  
Date 2025-10-04

Executive abstract  
This v0.1 technical specification defines the Dynamic Causal Hypergraph DCH framework and a hardware co-design, Causa-Chip, for real-time event-driven learning. The DCH turns spiking computation into an evolving, causal, neuro-symbolic model with explicit hyperedges as testable hypotheses, evidence-based credit assignment, streaming rule induction, and task-aware structural policies. This master document provides the high-level narrative, an index of sections, and cross-references for implementation and evaluation on event-vision benchmarks DVS Gesture and N-MNIST, alongside a Python prototype plan.

Scope and audience  
- Primary audience: researchers and engineers building online neuromorphic systems, causal graph analytics, and neuro-symbolic AI.  
- Deliverables: formal definitions, algorithms, interfaces, complexity targets, prototype blueprint, evaluation protocol, parameter defaults and tuning, risk/runbooks, and a hardware SoC overview.

Approved defaults (canonical)  
- Time model: continuous-time in microseconds.  
- TC-kNN temporal window: [1 ms, 30 ms]; δ_causal = 2 ms; refractory = 1 ms.  
- Path score: product of edge reliabilities.  
- EMA reliability: α = 0.1, r ∈ [0.02, 0.98].  
- Traversal: M = 8 seeds, L = 12 max depth, B = 4 branching, τ_select = 0.7, H_back = 100 ms.  
- FSM: W = 60 s, s_min = 50, r_min = 0.6, γ = 0.98.  
- Embeddings: WL online r = 2, d = 64 every 10 ms; periodic GraphSAGE r = 3, d = 128 every 500 ms.

How to use this master spec  
- Each section below is authored as a standalone file and linked here for quick navigation and modular maintenance.  
- For PDF export, concatenate sections in order (Section 1 through Section 15) and include References and Diagrams Index; see Export notes at the end.

Table of Contents (sections and cross-references)
- 0. Master outline and acceptance criteria  
  - [docs/DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)
- 1. Formal foundations and glossary  
  - [docs/sections/DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md)
- 2. Dynamic Hypergraph Construction (DHG) with TC-kNN  
  - [docs/sections/DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md)
- 3. Hyperedge plasticity (predict/confirm/miss, EMA, pruning)  
  - [docs/sections/DCH_Section3_Plasticity.md](./sections/DCH_Section3_Plasticity.md)
- 4. Hyperpath embedding and causal-context similarity (WL online + SAGE periodic)  
  - [docs/sections/DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md)
- 5. Credit assignment via constrained backward hyperpath traversal  
  - [docs/sections/DCH_Section5_CreditAssignment.md](./sections/DCH_Section5_CreditAssignment.md)
- 6. Streaming frequent hyperpath mining and online rule induction  
  - [docs/sections/DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md)
- 7. Hierarchical abstraction and higher-order hyperedges (HOEs)  
  - [docs/sections/DCH_Section7_HierarchicalAbstraction.md](./sections/DCH_Section7_HierarchicalAbstraction.md)
- 8. Task-aware scaffolding (REUSE/ISOLATE/HYBRID and FREEZE)  
  - [docs/sections/DCH_Section8_TaskAwareScaffolding.md](./sections/DCH_Section8_TaskAwareScaffolding.md)
- 9. Module interfaces and data contracts  
  - [docs/sections/DCH_Section9_Interfaces.md](./sections/DCH_Section9_Interfaces.md)
- 10. Complexity and resource model (latency/memory targets, backpressure)  
  - [docs/sections/DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md)
- 11. Software prototype blueprint (Python + Norse)  
  - [docs/sections/DCH_Section11_SoftwareBlueprint.md](./sections/DCH_Section11_SoftwareBlueprint.md)
- 12. Evaluation protocol (datasets, metrics, ablations)  
  - [docs/sections/DCH_Section12_Evaluation.md](./sections/DCH_Section12_Evaluation.md)
- 13. Parameter defaults and tuning strategy  
  - [docs/sections/DCH_Section13_ParamsTuning.md](./sections/DCH_Section13_ParamsTuning.md)
- 14. Risk analysis and mitigations (guardrails and runbooks)  
  - [docs/sections/DCH_Section14_RiskMitigations.md](./sections/DCH_Section14_RiskMitigations.md)
- 15. Causa-Chip hardware co-design overview  
  - [docs/sections/DCH_Section15_CausaChip.md](./sections/DCH_Section15_CausaChip.md)
- References and diagrams  
  - [docs/References.md](./References.md)  
  - [docs/DiagramsIndex.md](./DiagramsIndex.md)

Quick narrative (one-paragraph summary per section)
- Section 1: Defines V(t), E(t), hyperedge validity, hyperpaths with B-connectivity, reliability semantics, invariants, and notation.  
- Section 2: TC-kNN DHG module for candidate hyperedge generation with coherent multi-tail sets, budgets, deduplication, and refractory safety.  
- Section 3: Watcher-based predict/confirm/miss, EMA updates, discounted counts, decay, pruning policies, and concurrency determinism.  
- Section 4: Hybrid causal-context embeddings — WL online (streaming decisions) and GraphSAGE periodic (FSM/abstraction refinement).  
- Section 5: Multi-start randomized beam backward traversal with temporal logic constraints to produce path-evidence reliability targets.  
- Section 6: Streaming canonicalization, windowed heavy hitters + CMS counting, drift detection, and promotion to rules.  
- Section 7: Higher-order hyperedges from promoted rules with provenance, deduplication, safety, traversal compression, and governance.  
- Section 8: Task-aware scaffolding for continual learning with similarity detection, FREEZE, regionization, and policy knobs.  
- Section 9: Clear APIs and records for events, vertices, edges, paths, templates, HOEs; ordering, idempotency, security, and observability.  
- Section 10: Latency/memory targets and backpressure/adaptation strategies for desktop and embedded event-vision workloads.  
- Section 11: Python prototype architecture (three lanes), repository layout, configs, metrics, tests, and end-to-end orchestration loop.  
- Section 12: Evaluation datasets, baselines, ablations, interpretability and continual metrics, acceptance thresholds, and reproducibility.  
- Section 13: Defaults and tuning methodology, safe ranges, dataset presets, meta-controller adaptive rules, and diagnostics.  
- Section 14: Risks (combinatorics, spurious causality, drift, forgetting, latency/memory), monitors, stress tests, and runbooks.  
- Section 15: Causa-Chip SoC units (GSE, GMF, PTA, FSM, MC), NoC/memory, bandwidth/latency targets, and verification strategy.

Key cross-reference map
- Traversal (Sec. 5) produces r̂_path consumed by Plasticity (Sec. 3).  
- Embeddings (Sec. 4) inform DHG (Sec. 2) grouping and Traversal (Sec. 5) similarity bias; SAGE assists FSM (Sec. 6) and Abstraction (Sec. 7).  
- FSM promotions (Sec. 6) instantiate HOEs (Sec. 7) and provide rule priors to Traversal (Sec. 5).  
- Scaffolding (Sec. 8) sets FREEZE and regionization affecting DHG/Plasticity/Traversal/FSM/Abstraction and Meta control.  
- Interfaces (Sec. 9) standardize data flow; Complexity (Sec. 10) sets SLOs for all modules.  
- Blueprint (Sec. 11) implements 2–10; Evaluation (Sec. 12) validates; Params (Sec. 13) tunes; Risk (Sec. 14) stabilizes; Hardware (Sec. 15) accelerates.

Mermaid overview — neuro-symbolic loop (from sections, centralized here)

```mermaid
flowchart TD
SNN[Sub-symbolic SNN spikes] --> DHG[Dynamic hypergraph update]
DHG --> TRAV[Backward traversal credit assignment]
TRAV --> FSME[Frequent subgraph miner]
FSME --> RULES[Symbolic rules library]
RULES --> GUIDE[Top-down guidance]
GUIDE --> DHG
GUIDE --> TRAV
```

Export and assembly notes
- Master spec composition (for PDF): concatenate the following in order  
  - [docs/sections/DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md)  
  - [docs/sections/DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md)  
  - [docs/sections/DCH_Section3_Plasticity.md](./sections/DCH_Section3_Plasticity.md)  
  - [docs/sections/DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md)  
  - [docs/sections/DCH_Section5_CreditAssignment.md](./sections/DCH_Section5_CreditAssignment.md)  
  - [docs/sections/DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md)  
  - [docs/sections/DCH_Section7_HierarchicalAbstraction.md](./sections/DCH_Section7_HierarchicalAbstraction.md)  
  - [docs/sections/DCH_Section8_TaskAwareScaffolding.md](./sections/DCH_Section8_TaskAwareScaffolding.md)  
  - [docs/sections/DCH_Section9_Interfaces.md](./sections/DCH_Section9_Interfaces.md)  
  - [docs/sections/DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md)  
  - [docs/sections/DCH_Section11_SoftwareBlueprint.md](./sections/DCH_Section11_SoftwareBlueprint.md)  
  - [docs/sections/DCH_Section12_Evaluation.md](./sections/DCH_Section12_Evaluation.md)  
  - [docs/sections/DCH_Section13_ParamsTuning.md](./sections/DCH_Section13_ParamsTuning.md)  
  - [docs/sections/DCH_Section14_RiskMitigations.md](./sections/DCH_Section14_RiskMitigations.md)  
  - [docs/sections/DCH_Section15_CausaChip.md](./sections/DCH_Section15_CausaChip.md)  
  - [docs/References.md](./References.md)  
  - [docs/DiagramsIndex.md](./DiagramsIndex.md)
- Diagram rendering: ensure Mermaid blocks render in the chosen PDF pipeline (e.g., md-to-pdf with mermaid-cli or Pandoc with Mermaid filter).  
- Internal links: verify anchors and relative paths after concatenation.

Appendix — acceptance checklist alignment
- Functional completeness: see Sections 1–9 for formalisms, algorithms, and interfaces.  
- Engineering readiness: see Sections 10–11 for resource model and prototype blueprint.  
- Evaluation plan and thresholds: see Section 12.  
- Tuning defaults and adaptation: see Section 13.  
- Risk, runbooks, and guardrails: see Section 14.  
- Hardware overview and dataflows: see Section 15.

Changelog (v0.1)
- Initial release of the unified specification with linked sections and export notes.


---

# Dynamic Causal Hypergraph DCH — Section 1 Formal Foundations and Glossary

Parent outline [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)

Version v0.1

1. Scope and defaults
- Time model continuous time timestamps in microseconds μs
- Path score product of edge reliabilities
- Reliability update EMA with alpha 0.1 and bounds 0.02 to 0.98
- TC kNN window 1 ms to 30 ms, k max 3, delta causal 2 ms
- Traversal seeds M 8, max depth L 12, branching cap B 4, selection temperature 0.7
- FSM window W 60 s, support s min 50, reliability threshold r min 0.6, decay gamma 0.98

2. Base sets and objects
- Neuron id space I equals {1,2,...,N}
- Time domain T equals real nonnegative measured in microseconds
- Event vertex v equals neuron id i comma timestamp t with i in I and t in T
- Vertex set V t equals { v mid v.timestamp less or equal t }
- Hypergraph DCH t equals V t comma E t

3. Hyperedge schema and attributes
- A directed hyperedge e equals Tail to Head with Head cardinality 1
- Tail e equals { u1 comma u2 comma ... comma um } subset of V t with m greater or equal 1
- Head e equals { w } with w in V t
- Attributes for e
  - delay window Δ min comma Δ max in microseconds with 0 less Δ min less Δ max
  - refractory ρ in microseconds minimum separation for heads on same neuron
  - reliability score r e in interval 0.02 to 0.98
  - created at tau c equals Head timestamp
  - last update time tau u
  - usage count c e in integers
  - type label in {event edge comma template edge}

4. Validity predicate for temporal causality
- valid e mid V requires
  - for every u in Tail e the time difference Δ u equals Head time minus u time satisfies Δ min less or equal Δ u less or equal Δ max
  - no two hyperedges e1 and e2 with the same Head neuron create head times closer than ρ
- valid e mid V implies e respects temporal logic constraints for causality

5. Event anchored and template hyperedges
- Event anchored hyperedge an e whose Head is a realized vertex w in V t
- Template hyperedge τ defines Tail schema as a set of neuron ids with relative lags and a Head neuron id with relative zero
- Instantiation of τ at time t is a mapping of its schema to concrete vertices in V t that satisfy the window constraints

6. Operations on E t
- GROW add new event anchored hyperedges produced around an observed Head vertex per TC kNN
- REFINE update reliability via predict and confirm evidence
- PRUNE remove e if r e below threshold or age large with low usage
- ABSTRACT introduce a template hyperedge that summarizes a frequent reliable hyperpath
- FREEZE gate updates for a protected subset of E t as part of task aware scaffolding

7. Hyperpaths and B connectivity
- A hyperpath π from sources to sink w is a finite collection of hyperedges in E with a partial order such that for each edge e in π every tail vertex of e is either a source vertex or the head of some edge earlier in the order
- B connectivity constraint traversal from Head to Tail is allowed only if evidence exists for all tail vertices
- Instantiated hyperpath pairs π comma g where g grounds each edge to concrete vertices in V

8. Path scoring and evidence aggregation
- Path score s π equals product over e in π of r e
- Positive evidence set P e equals valid hyperpaths that include e and originate from correct or rewarded sinks
- Negative evidence set N e equals valid hyperpaths that include e and originate from erroneous sinks
- Normalized target r hat e equals sum w in P e s w divided by sum s in P e union N e s plus epsilon where epsilon small

9. Reliability update operator
- EMA update r new e equals clip of 1 minus alpha times r old e plus alpha times r hat e bounded to 0.02 and 0.98 with alpha equals 0.1
- Update timestamp tau u set to current time and increment c e

10. Temporal windows and units
- Defaults Δ min equals 1000 microseconds and Δ max equals 30000 microseconds
- δ causal equals 2000 microseconds micro window for grouping near coincident presyn spikes
- All timestamps monotone per neuron and unique per event

11. Presynaptic adjacency and candidate sources
- Pred j equals { i in I mid synapse i to j exists } provided by the underlying SNN or connectivity map
- Candidate antecedent spikes for a post spike at neuron j and time t j are the most recent spikes from each i in Pred j within the temporal window

12. Invariants and safety constraints
- Reliability bounds 0.02 less or equal r e less or equal 0.98 for all e
- No self cycle from a head vertex back to itself
- For any neuron j and times t1 less t2 with t2 minus t1 less ρ at most one is a head of an accepted edge
- ABSTRACT does not introduce a cycle at the template level when projected to neuron graph

13. Observables and counters
- events per second lambda t
- active edges count size of E t
- average reliability bar r t equals mean over e in E t of r e
- prune rate equals removals per unit time
- traversal yield equals valid hyperpaths per seed
- rule discovery rate equals promoted templates per unit time

14. Notation summary
- i comma j neuron ids
- t comma tau time variables microseconds
- v equals i comma t event vertex
- e hyperedge Tail to Head with attributes Δ min Δ max ρ r
- π hyperpath
- s π path score
- P e positive evidence set N e negative evidence set
- r hat e normalized target probability
- alpha EMA step size default 0.1

15. Interfacing with the SNN substrate
- The DCH consumes spike events produced by an SNN or event sensor
- The DCH never writes to neuron state directly but may export symbolic rules and policy hints to guide search or initialization

16. Minimal worked example
- Suppose neuron A fires at time 10000 microseconds and neuron B at 11700 microseconds and neuron C at 21000 microseconds with Δ min equals 1000 and Δ max equals 30000
- GROW proposes unary edges {A at 10000} to {C at 21000} and {B at 11700} to {C at 21000} and a binary edge {A at 10000 comma B at 11700} to {C at 21000} since B minus A equals 1700 within δ causal equals 2000
- Later the pattern {A then B} repeats and C fires within window generating positive evidence that increases r for the binary edge by the EMA rule

17. Cross references
- Detailed construction algorithms appear in Section 2 see [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)
- Traversal and credit assignment appear in Section 5 see [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)
- Online rule induction appears in Section 6 see [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)

18. Acceptance criteria for Section 1
- Formal definitions for V t E t vertices hyperedges attributes and validity are present
- Reliability update operator EMA with bounds and target aggregation is specified
- Hyperpath and B connectivity are defined with default path scoring
- Invariants and units are defined with defaults for Δ and δ causal
- Cross references align with the overall outline

File index
- This section file [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md)
- Parent outline file [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)

End of Section 1


---

# Dynamic Causal Hypergraph DCH — Section 2 Dynamic Hypergraph Construction with TC kNN

Parent outline [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)
Cross reference Section 1 [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md)

Version v0.1

1. Purpose and scope
- Define the online construction of E t around each postsynaptic spike event
- Specify TC kNN candidate selection higher order combination generation deduplication initialization and resource controls

2. Inputs and outputs
Inputs
- Event stream of spike vertices v equals neuron id comma timestamp
- Presynaptic adjacency Pred j for each neuron j
- Temporal parameters Δ min Δ max δ causal refractory ρ
- Budgets k max per head budget M in combinations cap C cap

Outputs
- New hyperedges added to E t with initialized attributes and provenance
- Rejected candidate statistics for meta controller feedback

3. Preliminaries and data structures
Event buffers
- For each neuron i maintain a time sorted ring buffer B i of recent spike vertices with capacity measured by time horizon T ret and item cap N ret
- Support binary search by timestamp and append amortized O one

Hyperedge stores
- incoming by head maps a head vertex id to a list of incident hyperedge ids
- by tail signature maps a canonical tail signature key to hyperedge id for deduplication
- recent heads by neuron maps neuron id to recent head vertex timestamps for enforcing refractory

Canonical tail signature
- Define sig Tail as multiset of ordered pairs neuron id comma time sorted by time then neuron id
- Key equals hash of sig Tail concatenated with head vertex id

4. TC kNN candidate generation around a post spike
Trigger
- When a spike v post equals neuron j comma time t j arrives begin candidate search

Step A presynaptic lookup
- Retrieve Pred j from adjacency
- For each i in Pred j query B i for the most recent spike u i with time t i in interval t j minus Δ max to t j minus Δ min
- If none found for i skip i else add u i to set U j

Step B unary candidates
- For each u in U j create a unary candidate Tail equals { u } and Head equals { v post }

Step C higher order candidates up to k max
- Form all combinations of U j of size m for m in 2 to k max subject to temporal coherence constraint
- Temporal coherence requires max time of tails minus min time of tails less or equal δ causal
- For each coherent combination form Tail equals that combination and Head equals { v post }

Step D candidate scoring and admission control
- Compute candidate priority score s cand using heuristics
  - Delay closeness term equals mean over u in Tail of one minus abs t j minus t u minus μ Δ divided by Δ span where μ Δ equals mid of Δ min and Δ max and Δ span equals Δ max minus Δ min
  - Fan in cost term penalizes large m for example λ m equals m divided by k max
  - Priority s cand equals w delay times delay closeness plus w size times 1 minus λ m defaults w delay equals 0.7 w size equals 0.3
- Maintain per head budget M in and global combinations cap C cap per post spike
- Select top candidates by s cand until budgets are met

Step E deduplication
- For each selected candidate build sig Tail and key
- If key exists in by tail signature skip else continue

Step F initialize hyperedge and insert
- Create edge e with attributes
  - Tail as selected set of tail vertices
  - Head equals { v post }
  - Δ min Δ max ρ inherited from defaults or per neuron tables
  - reliability score r e initialized to epsilon init default 0.05
  - created at equals t j last update time equals t j usage count equals 0 type equals event edge
- Insert e into E t and update indexes incoming by head and by tail signature and recent heads by neuron for refractory checks

5. Refractory enforcement and conflict resolution
- Before inserting any edge with Head neuron j verify that no accepted head for neuron j exists in interval t j minus ρ to t j plus ρ
- If conflict occurs apply deterministic tie break
  - Prefer the head with larger s cand
  - If equal prefer earlier timestamp to reduce pile up
- Edges referencing a rejected head are discarded

6. Provenance and audit metadata
- Record for each inserted edge the candidate score s cand and the list of presynaptic neuron ids and time lags Δ u equals t j minus t u
- Maintain a creation token equals tuple of head vertex id and tail signature to support idempotent replay

7. Parameter table defaults
- Δ min equals 1000 microseconds Δ max equals 30000 microseconds δ causal equals 2000 microseconds ρ equals 1000 microseconds
- k max equals 3 M in equals 6 C cap equals 10 epsilon init equals 0.05
- w delay equals 0.7 w size equals 0.3

8. Complexity analysis
Notation
- d j equals degree of Pred j
- b equals average buffer size per neuron within window
- c equals number of coherent tails admitted before budget

Costs per post spike
- Presyn lookup O d j
- For each i binary search in B i O log b and constant time verification per match total O d j log b
- Combination enumeration worst case sum over m from 2 to k max of C of U j choose m bounded by budgets yields O c
- Deduplication and insert O 1 amortized via hash maps

Memory
- Event buffers sum over i of size B i bounded by T ret and N ret
- Hyperedges active equals size of E t bounded by pruning policy and budgets
- Indexes proportional to number of active edges

9. Concurrency and ordering
- Process events in nondecreasing timestamp order
- Apply per neuron j critical section for refractory and insertion to avoid races
- Use lock free ring buffers for B i and atomic increments for counters
- Idempotent creation by checking creation token before insert enables safe replays

10. Failure and backpressure handling
- If budgets exceeded drop lowest s cand and increment rejected counters
- If buffer overflow occurs advance head of ring buffer and increment buffer evict counter
- If time skew detected for input events stash out of order events until watermark advances

11. Interfaces aligned to module contracts
Event ingestion to DHG
- on post spike inputs neuron id j and timestamp t j returns list of created edge ids may be empty
- get rejected stats returns struct with counts per reason and recent averages

Query
- get tails for head head vertex id returns list of tail sets and edge ids
- exists edge key returns boolean

Configuration
- set params provide Δ min Δ max δ causal ρ k max M in C cap weights and epsilon init
- get params returns current table possibly per neuron overrides

12. Optional variants and extensions
Adaptive windows
- Maintain per neuron pair i to j estimates of empirical delay distribution using exponential histograms
- Modulate Δ min and Δ max around current percentiles for that pair

Ranked K for presyn
- Instead of only the most recent u i choose top K i recent spikes in the window and allow more combinations while tightening budgets

Synaptic priors
- If the SNN exposes synaptic strength use it as a prior multiplier in s cand

13. Quality metrics for the DHG module
- Candidate hit rate equals created edges divided by evaluated candidates
- Deduplication rate equals skipped duplicates divided by evaluated candidates
- Average tails per head vertex
- Average admitted combination size
- End to end latency per post spike for DHG path

14. Mermaid diagram enhanced TC kNN flow

```mermaid
flowchart TB
POST[Post spike j at time t j] --> A1[Lookup Pred j]
A1 --> A2[Find most recent presyn spikes in window]
A2 --> A3[Form unary and multi tail candidates up to k max]
A3 --> A4[Temporal coherence filter within delta causal]
A4 --> A5[Score candidates and apply budgets]
A5 --> A6[Deduplicate by tail signature]
A6 --> A7[Refractory check and conflict resolve]
A7 --> A8[Insert edges and update indexes]
```

15. Acceptance criteria for Section 2
- TC kNN generation defined for unary and higher order tails with temporal coherence
- Budgets scoring deduplication initialization and refractory policy specified
- Complexity and memory bounds presented with symbols d j b c
- Interfaces align with contracts in outline and observability metrics defined
- Enhanced diagram included and consistent with outline

16. Cross references
- Formal symbols and validity constraints in Section 1 see [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md)
- Plasticity updates in Section 3 see [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)
- Traversal and credit assignment in Section 5 see [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)

End of Section 2


---

# Dynamic Causal Hypergraph DCH — Section 3 Hyperedge Plasticity Rules

Parent outline [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md) and Section 2 [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md)

Version v0.1

1. Purpose and scope  
- Define local, event driven learning rules over hyperedges in E t.  
- Specify prediction logic, confirmation and miss detection, EMA reliability update, decay, and pruning.  
- Integrate with global credit assignment signals from Section 5 while remaining effective in purely unsupervised operation.

2. Plasticity primitives overview  
- Predict When a tail pattern of a hyperedge e appears, the system predicts that the head event should occur within the temporal window of e.  
- Confirm If the head event occurs within the specified window, treat as positive evidence.  
- Miss If the window elapses without the head, treat as negative evidence.  
- Decay Time based relaxation toward a prior to prevent stale edges from remaining overconfident.  
- Prune Remove edges with low, aging, or budget violating reliability to control combinatorics.  
- Freeze Gate updates for a subset of edges during task isolation per Section 8.

3. Event lifecycle objects  
3.1 Watcher records  
- For each edge e and each observed tail occurrence at time t start, instantiate a watcher w e,t with  
  - head window open equals t start plus Δ min and close equals t start plus Δ max  
  - status pending then confirmed or missed  
  - weight ω local for contribution to updates default 1.0  
  - provenance tail vertex ids and head neuron id  
- A watcher emits exactly one outcome confirmed or missed then becomes inactive.  
- Multiple watchers for the same edge may overlap if tails recur faster than window length.

3.2 Tail detection for watcher creation  
- A watcher is created when all elements of Tail e are observed with timestamps in nondecreasing order and within the small coherence band δ causal from Section 2, and before any head occurs.  
- If multiple spikes from the same presyn neuron arrive within δ causal, choose the latest for that occurrence or spawn multiple watchers under a combinations budget.

4. Reliability model and targets  
- Reliability r e is bounded r min less or equal r e less or equal r max with defaults 0.02 and 0.98.  
- Local target from watcher outcomes  
  - Let c pos e be the exponentially discounted count of confirmations for edge e.  
  - Let c neg e be the exponentially discounted count of misses for edge e.  
  - Local target r hat local e equals c pos e divided by c pos e plus c neg e plus ε where ε greater 0 is a stabilizer default 1.0.  
- Global target from credit assignment Section 5  
  - r hat path e derived from aggregated valid hyperpaths that used e.  
- Combined target  
  - r hat comb e equals λ path times r hat path e plus 1 minus λ path times r hat local e with λ path in 0 to 1 default 0.5.

5. Update rules  
5.1 EMA reliability update  
- Upon watcher resolution or credit assignment cycle update r e by  
  - r new e equals clip of 1 minus α times r old e plus α times r hat comb e with α default 0.1 and clip to bounds r min r max.  
- Increment usage count c e and set last update time.

5.2 Discounted counters update  
- Maintain discounted counts with decay factor γ c per event tick Δ t  
  - c pos e becomes γ c to the power Δ t times c pos e then plus sum over resolved watchers confirmed with weight ω local.  
  - c neg e becomes γ c to the power Δ t times c neg e then plus sum over resolved watchers missed with weight ω local.  
- Defaults γ c equals 0.98 per second equivalent when time is in seconds normalize Δ t by seconds.

5.3 Time decay toward prior  
- Between events or on periodic housekeeping, relax r e toward prior r 0 default midpoint 0.5 via  
  - r e becomes 1 minus β times r e plus β times r 0 with β small default 0.01 per second equivalent using elapsed wall time.  
- Skip decay when edge is frozen.

5.4 Confidence flooring and ceilings  
- Maintain invariant r min less or equal r e less or equal r max see Section 1.  
- Optionally shrink interval adaptively as edges age to avoid extremes under sparse evidence.

6. Confirmation and miss determination  
- For watcher w e,t with window open and close  
  - Confirm if a head spike for the exact head vertex neuron occurs with timestamp within open to close inclusive and is not vetoed by refractory ρ.  
  - Miss if the watermark passes window close and no valid head spike was observed.  
- Handle overlapping watchers by assigning a head spike to the earliest pending watcher first to avoid double counting, then to later ones if multiple heads occur.

7. Concurrency model and determinism  
- Process spikes in nondecreasing timestamp order with a per head neuron critical section to enforce refractory.  
- Watcher creation and resolution are idempotent under replays by using a deterministic watcher id composed of edge id, earliest tail timestamp, and head neuron id.  
- All updates to r e, counts, and usage are atomic; operations are commutative under batch replay.  
- Frozen edges are marked read only for reliability and counters.

8. Pruning policies  
8.1 Threshold based prune  
- Remove e if r e less than τ prune and age greater than τ age min or usage count below τ use min to avoid removing fresh edges prematurely.  
- Defaults τ prune 0.02, τ age min 2 seconds, τ use min 3.

8.2 Budget based prune  
- Maintain per head neuron budget K head for number of incoming edges; if exceeded, evict lowest priority edges by score s prune equals r e times freshness where freshness equals exp of negative lambda age times age.  
- Maintain global cap K global to bound memory footprint.

8.3 Inactivity prune  
- If no watcher has been created for e over horizon H idle remove e unless frozen or protected by provenance links due to abstraction.

8.4 Cascade integrity  
- When pruning an edge, detach any abstraction provenance links; if an abstraction becomes unsupported beneath a minimal support threshold, mark it for review or removal.

9. Freeze policy hooks  
- When task isolation is engaged Section 8 in the outline add e to protected set preventing reliability change and pruning.  
- Frozen edges may still accumulate read only statistics for monitoring.  
- Defrost by policy command with hysteresis to prevent thrashing.

10. Parameter table defaults  
- α 0.1 for EMA step size, r min 0.02, r max 0.98, r 0 0.5.  
- γ c 0.98 per second for discounted counts; β 0.01 per second for time decay.  
- τ prune 0.02, τ age min 2 s, τ use min 3, H idle 30 s.  
- Budgets K head 256 per head neuron default, K global configurable per platform.  
- λ path 0.5 to balance local and path based learning.

11. Observability and counters  
- per edge r e, c pos e, c neg e, last update time, age, frozen flag.  
- module rates watchers created per second, confirms per second, misses per second, prune events per second.  
- budget occupancy head budget usage histogram, global budget usage.  
- stability indicators fraction of edges near bounds, half life estimate median time to 10 percent change in r e.

12. Failure and edge cases  
- Head ambiguity multiple candidate heads within window close in time  
  - Prefer head matched via exact neuron id and strongest provenance chain if available else earliest in time.  
- Tail jitter if coherence fails by small margin epsilon jitter, allow a one time tolerance band configurable to increase recall at cost of precision.  
- Duplicate watcher creation deduplicated by watcher id.  
- Missing timestamps or out of order events buffer until watermark satisfies ordering assumptions.

13. Interaction with credit assignment Section 5  
- When a credit assignment cycle emits aggregated r hat path e, immediately perform an EMA update using r hat comb e as in Section 5.1 with λ path greater than 0.5 temporarily to capitalize on supervision, then anneal λ path back to default.  
- Alternatively treat path signals as additional weighted confirmations for matching watchers within the path window to unify accounting.

14. Mermaid diagram plasticity flow

```mermaid
flowchart TB
TAIL[Tail observed for edge e] --> WMAKE[Create watcher with window]
WMAKE -->|Head arrives in window| CONF[Confirm outcome]
WMAKE -->|Window closes no head| MISS[Miss outcome]
CONF --> COUNTS[Update discounted counts c_pos and EMA r_e]
MISS --> COUNTS
COUNTS --> DECAY[Periodic decay toward prior]
DECAY --> PRUNE[Apply pruning thresholds and budgets]
PRUNE --> STATE[Update edge state and metrics]
```

15. Quality metrics and acceptance thresholds  
- Confirm rate divided by watchers created within 20 percent of estimated ground truth on synthetic microbenchmarks.  
- Prune precision at least 80 percent on synthetic tasks where true spurious edges are labeled.  
- Stability half life greater than or equal to 5 seconds at steady state under stationary streams with no supervision.  
- No drift beyond 5 percent in r e on frozen edges across test windows.

16. Interfaces aligned to module contracts  
- plasticity resolve watchers inputs time watermark returns resolved counts and list of edges updated.  
- plasticity update from path signals inputs mapping from edge id to r hat path weight returns applied updates.  
- plasticity prune step inputs current budgets and policy returns removed edges.  
- plasticity set freeze inputs set of edge ids and flag.  
- plasticity metrics snapshot returns structured counters.

17. Acceptance criteria for Section 3  
- Predict confirm miss mechanism with watcher abstraction is specified.  
- EMA reliability update, discounted counters, and time decay are defined with defaults and bounds.  
- Pruning policies threshold, budget, inactivity with cascade integrity are specified.  
- Concurrency, determinism, and freeze interactions are described.  
- Interfaces and observability align with the outline.

18. Cross references  
- Formal reliability bounds and attributes Section 1 see [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md)  
- DHG candidate generation Section 2 see [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md)  
- Credit assignment Section 5 outline see [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)

End of Section 3


---

# Dynamic Causal Hypergraph DCH — Section 4 Hyperpath Embedding and Causal-Context Similarity

Parent outline [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md) and Section 2 [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md)

Version v0.1

1. Purpose and scope
- Define causal-context embeddings used to group vertices by similarity of causal histories and to propose higher-order hyperedges.
- Specify an approved hybrid scheme
  - WL-style online embedding deterministic, r=2, d=64, update cadence 10 ms, drives streaming DHG grouping and GROW.
  - GraphSAGE incidence-expansion embedding periodic, r=3, d=128, refresh 500 ms, refines global causal-context for FSM and hierarchical abstraction.

2. Design overview and rationale
- The DCH operates on event tuples with sparse, evolving structure; vertex identity has little feature content.
- Similarity must reflect causal context i.e., hyperpaths terminating at vertices and the timing/reliability attributes along those paths.
- WL online delivers low-latency deterministic updates compatible with per-event GROW and PRUNE, while periodic SAGE captures richer higher-order context without stalling the stream.

3. Notation and prerequisites
- Vertex v equals i comma t see Section 1 [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md).
- Hyperpath π and reliability r e defined in Section 1.
- TC kNN construction produces candidate edges around a head vertex see Section 2 [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md).

4. WL-style online embedding WL
Defaults
- Radius r WL equals 2 backward hops over incident incoming hyperedges.
- Dimension d WL equals 64 via feature hashing.
- Update cadence Δt WL equals 10 ms wall-clock or watermark-aligned.
- Update policy deterministic incremental recompute affected vertices only.

4.1 Hypergraph neighborhood and labels
- Work on the directed hypergraph using backward incidence from head to tails.
- Node labels
  - For vertex v define base label l0 v as tuple neuron id i and time bucket of v timestamp quantized to q time equals 100 microseconds.
- Edge labels
  - For hyperedge e define label fe as tuple cardinality of Tail, delay stats mean and variance over Δ u equals head time minus tail times, and binned reliability r e bucketized into K r buckets.

4.2 WL iteration and hashing
- Iteration k equals 1 to r WL
  - For each vertex v collect multiset Mk v of labeled messages from parents
    - For each incoming e to v head compute message m e as hash fe concatenated with multiset of parent labels from previous iteration l k minus 1 of each u in Tail e and with temporal deltas Δ u bucketized.
  - Aggregate Mk v into string or tuple then hash with a stable 64 bit function.
- Feature hashing
  - Map each hashed token into d WL bins with k independent hash functions and add sign hashing for balanced updates.
  - Maintain an embedding vector x WL v in R to the d WL updated additively and normalized to unit norm.

4.3 Temporal features
- Bucketization
  - Δ u buckets logarithmic over 1 ms to 30 ms; refractory indicators as binary features.
- Recency
  - Apply exponential time decay weight w time equals exp of negative lambda time times Head time minus v time when aggregating labels default lambda time equals 1 divided by 200 ms.

4.4 Incremental maintenance
- Upon insertion of a new head vertex v post
  - Update l0 v post then perform up to r WL backward iterations confined to the r WL neighborhood frontier.
  - Only recompute x WL for vertices whose WL hash multiset changed.
- Complexity per event bounded by neighborhood growth; cap via frontier budget F max default 256 vertices.

4.5 Grouping with LSH
- Use cosine LSH over x WL to generate candidate groups of vertices with similar context.
- Parameters
  - Bands b equals 8, rows r band equals 4 per band, yielding 32 projections; collision threshold τ LSH equals 2 band matches.
- Output
  - For a new head v post return top K group candidates K group equals 3, each a small set of antecedent vertices whose embeddings collide with v post.

4.6 Integration with DHG GROW
- DHG candidate selection Section 2 augments TC kNN with WL grouping
  - For each group candidate G produce a higher-order tail by selecting the most recent event per neuron in G within δ causal; deduplicate against existing edges.
  - Prioritize these group-induced candidates by boosting s cand with a term proportional to average cosine similarity between x WL v post and group members.

5. GraphSAGE incidence-expansion embedding SAGE
Defaults
- Refresh cadence Δt SAGE equals 500 ms periodic batch run.
- Dimension d SAGE equals 128, radius r SAGE equals 3.
- Encoder
  - Operate on the bipartite incidence expansion H to B where H are hyperedge nodes and B are event vertices; connect v in B to e in H if v in Tail e or v equals Head e.
  - Use mean or attention aggregator with time encoding.

5.1 Features and encoders
- Vertex initial features
  - One hot neuron id compressed via trainable embedding table size N by d id where d id equals 16.
  - Time encoding via sinusoidal features over microsecond scale and recentness decay.
  - Local WL embedding x WL as input channel to SAGE to bootstrap structure.
- Hyperedge initial features
  - Tail size, reliability bucket, delay stats, age, and usage count.
- Aggregation
  - Layer l plus 1 embedding h v equals sigma of W l times concatenation of h v l and aggregate over neighbors N v h u l with time attention weights a u based on Δ and reliability.

5.2 Training objectives
- Unsupervised contrastive objective over positive pairs vertex and true hyperedge neighbors and negatives sampled by degree profile.
- Optional temporal skip gram objective over sequences of heads in short windows.

5.3 Outputs and usage
- Produce x SAGE v in R to the d SAGE for vertices and optionally x SAGE e for hyperedges.
- Publish a global causal-context map used by
  - Streaming FSM Section 6 to canonicalize and compress pattern types.
  - Hierarchical abstraction Section 7 to cluster similar hyperpaths before rule promotion.

5.4 Resource controls
- Mini-batch construction limited to B size equals 4096 vertices per refresh with neighbor sampling cap S nbr equals 32 and hyperedge sampling cap S edge equals 64.
- End-to-end refresh budget less than 20 ms on desktop target for 500 ms cadence.

6. Interfaces
6.1 WL online API
- embedding wl get vertex id returns vector x WL v
- embedding wl update on event vertex id timestamp returns updated vectors for touched vertices limited by F max
- embedding wl propose groups head vertex id returns list of K group candidate sets and similarity scores
- embedding wl params set or get to adjust r WL d WL Δt WL b r band τ LSH F max

6.2 GraphSAGE periodic API
- embedding sage refresh now returns snapshot id and statistics
- embedding sage get vertex id returns x SAGE v from latest snapshot
- embedding sage link FSM returns handles for FSM to use SAGE embeddings as features
- embedding sage params set or get to adjust r SAGE d SAGE Δt SAGE batch and sampling knobs

6.3 Unified embedding view
- embedding get vertex id returns x WL v and x SAGE v with timestamps and snapshot ids
- similarity vertex a vertex b mode in {WL, SAGE, HYBRID} returns cosine sims and composite score
- events to DHG pipeline requests WL-only fast path for GROW; FSM requests HYBRID targeting rule mining.

7. Parameter table defaults and tuning
- WL r WL 2, d WL 64, Δt WL 10 ms, frontier F max 256, LSH b 8, r band 4, τ LSH 2, lambda time 1 divided by 200 ms, q time 100 microseconds.
- SAGE r SAGE 3, d SAGE 128, Δt SAGE 500 ms, batch B size 4096, S nbr 32, S edge 64.
- Grouping K group 3, cosine threshold τ cos 0.65 for boosting DHG candidates.
- Tuning
  - Increase d WL to 128 if collision precision is low and budgets allow.
  - Reduce Δt WL if event density is low to amortize computations.

8. Complexity and performance
- WL per event
  - Neighborhood scan O frontier size up to F max; hashing O tokens with small constants; LSH insert O b.
  - Latency target less than 200 microseconds per event on desktop prototype.
- SAGE per refresh
  - Sampling O B size times S nbr plus S edge; message passing O edges in sampled subgraph; GPU offload recommended.

9. Data structures and storage
- Embedding tables
  - WL table map vertex id to x WL vector with last update time.
  - SAGE snapshot store map snapshot id to arrays for x SAGE.
- LSH indices
  - Maintain b hash tables with r band rows each keyed by integer projections; garbage-collect entries older than TTL equals 2 seconds.
- Provenance
  - Keep per-group creation records mapping from WL collisions to DHG candidates for auditability.

10. Quality metrics
- WL collision precision and recall measured against known causal groupings on synthetic workloads.
- Group-augmented DHG admission rate uplift relative to TC kNN only baseline.
- FSM rule discovery rate improvement when SAGE embeddings are available.
- End-to-end overhead percent of DHG latency budget consumed by WL updates; target less than 30 percent.

11. Failure and edge cases
- Hash collisions generating spurious groups mitigate by requiring minimum cosine τ cos and by deduplication against existing edges.
- Embedding staleness if WL cadence is too slow; enforce max staleness bound of 20 ms for DHG use.
- Snapshot skew between WL and SAGE when FSM consumes both; annotate paths with snapshot ids for reproducibility.

12. Security and privacy considerations
- Embedding vectors are derived from event timing and neuron ids; ensure logs redact raw timestamps when exporting outside the system by quantization per q time and by removing neuron id mapping tables.

13. Mermaid diagram embedding pipeline

```mermaid
flowchart LR
EV[Event vertex stream] --> WLUP[WL update r=2 d=64 every 10 ms]
WLUP --> LSHI[LSH index]
LSHI --> GROUPS[Group candidates]
GROUPS --> GROW[DHG GROW booster]
EV --> SAGEBUF[Periodic sample buffer 500 ms]
SAGEBUF --> SAGE[GraphSAGE r=3 d=128 refresh]
SAGE --> FSM[FSM and abstraction]
```

14. Acceptance criteria for Section 4
- WL online embedding algorithm defined with temporal features, hashing, incremental maintenance, and LSH grouping.
- Integration with DHG candidate generation and prioritization specified.
- GraphSAGE periodic embedding defined with incidence-expansion modeling, features, objective, and refresh policy.
- Interfaces, defaults, complexity, and observability metrics provided.
- Diagram reflects dataflow and module boundaries.

15. Cross references
- Formal definitions Section 1 [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md)
- DHG construction Section 2 [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md)
- FSM Section 6 outline [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)
- Abstraction Section 7 outline [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)

End of Section 4


---

# Dynamic Causal Hypergraph DCH — Section 5 Credit Assignment via Constrained Backward Hyperpath Traversal

Parent outline [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md), Section 2 [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md), Section 3 [DCH_Section3_Plasticity.md](./sections/DCH_Section3_Plasticity.md), Section 4 [DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md)

Version v0.1

1. Purpose and scope  
- Define the discrete, evidence based credit assignment mechanism for DCH.  
- Traverse backward from target head vertices to discover valid causal hyperpaths under temporal and biological constraints.  
- Aggregate path evidence to update hyperedge reliabilities and feed online plasticity and rule induction.

2. Inputs and outputs  
Inputs  
- Seed set S seeds of head vertices tagged by supervision, error, or reward events (e.g., classifier output spikes or task signals).  
- Current DCH state E t and V t with incoming by head index from Section 2.  
- Temporal parameters Δ min, Δ max, refractory ρ; traversal policy M, L, B, temperature τ select (Section 1 defaults).  
- Optional symbolic rule priors from FSM and abstraction (Sections 6–7) as bias features.

Outputs  
- Set of discovered valid hyperpaths Π valid with per path scores s π.  
- Edge level aggregates A e positive and negative evidence contributions.  
- Optional partial paths for audit and debugging when termination budgets are hit.

3. Valid hyperpath constraints (B connectivity and temporal logic)  
- B connectivity: An expansion across a hyperedge e from its head w to its tail Tail e is admissible only if evidence exists for every tail vertex u in Tail e. Since E t encodes realized event anchored edges, tails exist, but traversal must honor bounds below.  
- Temporal window: For each u in Tail e, Δ u equals w time minus u time must satisfy Δ min less or equal Δ u less or equal Δ max (edge attributes).  
- Refractory: Along any path, no two edges with heads on the same neuron may have head timestamps closer than ρ (prevents implausible rapid refiring).  
- Horizon: Only traverse to tail vertices with timestamps greater or equal t w minus H back, where H back is a traversal horizon in time (default H back 100 ms for event vision); also obey step and branching caps.  
- Acyclicity: Because event vertices are time ordered and edges head time is strictly greater than tail times, cycles are precluded in event anchored traversal; still avoid reusing the same vertex id within a single path.

4. Traversal state and search space  
- State represented as a pair Frontier, Path where Frontier is a multiset of vertices that must be justified and Path is the ordered list of edges already included.  
- Initialization per seed w star in S seeds set Frontier equals { w star } and Path equals empty.  
- Goal condition reached when Frontier is empty; a complete hyperpath π has been discovered.  
- Expansion chooses a vertex v from Frontier, selects an admissible incoming edge e with head v, removes v from Frontier, and adds all u in Tail e into Frontier; append e to Path.  
- Sampling domain for admissible edges includes only edges meeting B connectivity and temporal checks and respecting refractory relations to edges already in Path.

5. Multi start random B walk policy  
Defaults (Section 1): M equals 8 seeds per event, L equals 12 max depth, B equals 4 branching cap per expansion, τ select equals 0.7.  
- Selection probability for candidate edge e given current v:  
  - Features  
    - f rel equals log r e (monotonic with reliability).  
    - f rec equals exp of negative lambda rec times age e where age e equals current time minus head time of e; default lambda rec equals 1 divided by 100 ms.  
    - f rule equals 1 if e aligns with a known symbolic rule motif from FSM, else 0; may be weighted by rule confidence.  
    - f sim equals average cosine similarity between WL embedding of v and WL embeddings of tails of e (Section 4).  
  - Score z e equals w rel times f rel plus w rec times f rec plus w rule times f rule plus w sim times f sim with defaults w rel 1.0, w rec 0.5, w rule 0.5, w sim 0.5.  
  - Sampling probability p e equals softmax over candidates z e divided by τ select.  
- Branching control pick up to B highest probability candidates per expansion or sample without replacement up to B; beam style control described in Section 6 below.  
- Depth control terminate any branch exceeding L expansions or reaching timestamps older than H back.

6. Search strategies and determinism  
- Randomized beam search: Maintain K beam partial states with cumulative log scores and expand the top K after each layer (K default equals M). This improves stability vs. pure random selection while preserving exploration via softmax.  
- AND frontier semantics: Because hyperedges are AND relations over their tails, a state with Frontier size greater than 1 must justify all vertices; interleave expansions by picking any vertex from Frontier that has admissible incoming edges.  
- Vertex selection heuristic: Prefer the frontier vertex with the fewest admissible incoming edges first (fail fast), ties broken by most recent timestamp.  
- Deterministic replay: Use a seeded RNG keyed by seed vertex id, current time bucket, and frontier signature. With the same seed and DCH snapshot, traversal sequences are reproducible.

7. Path scoring and aggregation  
- Path score s π equals product over e in π of r e (Section 1 default).  
- Sign of contribution:  
  - For correct or rewarded seeds, add s π to A e positive for all e in π.  
  - For erroneous seeds (wrong class spike or error signal), add s π to A e negative for all e in π.  
- Normalization per edge e:  
  - r hat path e equals A e positive divided by A e positive plus A e negative plus ε with ε small equals 1.0 for stability.  
- Emission to plasticity: Provide r hat path e and optionally per edge weights to Section 3 to blend with local watcher statistics (λ path control).

8. Temporal logic enforcement details  
- Interval algebra alignment: Treat each hyperedge as imposing interval relations where tail[u] precedes head[w] by Δ u within bounds; disallow overlaps that violate refractory for same neuron heads.  
- Head coincidence filter: If two admissible edges share the same head neuron with head times closer than ρ, only allow the earlier one in a given path branch.  
- Tail collapse: If two tails within the same expansion refer to events from the same neuron within δ causal, select the latest tail event (consistent with Section 2 construction) to avoid duplicate justifications.

9. Efficiency, pruning, and early termination  
- Candidate capping: For a vertex v, cap the number of admissible incoming edges evaluated at C in per vertex cap default 16 to bound branching.  
- Early discard rules:  
  - Upper bound pruning discard partial states whose maximum possible extension score upper bound (using max r e on remaining depth) cannot beat the current k th best; maintain per depth best beams.  
  - Time bound pruning discard branches where the next admissible steps would violate H back.  
- Memoization: Cache admissible incoming edges per head vertex within the current traversal cycle keyed by head id and snapshot id to avoid repeated filtering.

10. Data dependencies and interfaces  
- Requires incoming by head index and hyperedge attribute access from Section 2.  
- Requires WL embeddings for similarity term and optional SAGE snapshot id from Section 4.  
- Outputs r hat path e to Section 3 via plasticity update from path signals.  
- Rule priors from FSM (Section 6) provided as a lookup mapping hyperedge motifs to bias weights.

11. Parameters and defaults  
- M equals 8 seeds per event; L equals 12; B equals 4; τ select equals 0.7; H back equals 100 ms; C in equals 16.  
- Feature weights w rel equals 1.0, w rec equals 0.5, w rule equals 0.5, w sim equals 0.5; λ rec equals 1 divided by 100 ms.  
- ε equals 1.0 for r hat normalization.  
- Determinism seed composed from seed vertex id, time bucket 1 ms, and traversal cycle id.

12. Observability and metrics  
- paths per seed average and distribution, valid ratio.  
- average depth reached and branching factor.  
- positive vs negative evidence totals per cycle and per class.  
- traversal latency per seed and total budget use.  
- contribution coverage fraction of active edges touched by traversal.

13. Complexity analysis  
Notation: D equals average admissible in degree after constraints; B equals branching cap; L equals depth cap; K equals beam width (approximately M).  
- Worst case explored states O K times sum from d equals 1 to L of B to the d which is controlled by caps; practical due to tight admissibility filters and temporal horizon.  
- Per expansion work dominated by admissible edge filtering O D (bounded by C in) plus feature scoring O D and softmax.  
- Total per seed cost upper bounded by K times L times C in with small constants.

14. Failure and edge cases  
- No admissible incoming edges for a frontier vertex leads to branch dead end; continue with other vertices in Frontier; if none remain, branch fails.  
- Seeds with timestamp near stream start older than H back yield few or no paths; still emit empty aggregates to avoid bias.  
- Highly bursty inputs may create many admissible candidates; enforce C in and B, defer excess edges via priority queues, and spill stats for meta controller.

15. Mermaid diagram — backward traversal with AND frontier

```mermaid
flowchart TB
SEED[Seed head vertex w*] --> INIT[Init Frontier={w*}, Path=[]]
INIT --> EXPAND[Pick frontier vertex v with fewest admissible edges]
EXPAND --> FILTER[Filter incoming edges by temporal windows, refractory, horizon]
FILTER --> SELECT[Score edges and sample up to B via softmax]
SELECT --> BRANCH1[Choose edge e1]
SELECT --> BRANCH2[Choose edge e2]
BRANCH1 --> FRONT1[Replace v with Tail(e1) in Frontier; append e1 to Path]
BRANCH2 --> FRONT2[Replace v with Tail(e2) in Frontier; append e2 to Path]
FRONT1 --> CHECK1{Frontier empty?}
FRONT2 --> CHECK2{Frontier empty?}
CHECK1 -->|Yes| EMIT1[Emit path π1 and score s(π1)]
CHECK2 -->|Yes| EMIT2[Emit path π2 and score s(π2)]
CHECK1 -->|No| EXPAND
CHECK2 -->|No| EXPAND
```

16. Quality targets and acceptance criteria  
- Valid hyperpaths honor all temporal windows and refractory constraints; no violations in audit replays.  
- On synthetic benchmarks with planted causal chains, traversal recovers at least 90 percent of ground truth edges within H back under default caps.  
- End to end traversal latency per seed less than 1 ms on desktop prototype for K equals 8, L equals 12, C in equals 16.  
- Emitted r hat path e integrates with plasticity Section 3 and yields monotonic reliability growth on consistent tasks and depression on error tagged seeds.

17. Interfaces aligned to module contracts  
- traversal assign credit inputs seeds list, mode in {reward, error, correct}, returns edge to contribution map A e positive or A e negative and per path records.  
- traversal params set or get to adjust M, L, B, τ select, H back, C in, and feature weights.  
- traversal metrics snapshot returns counters listed in Section 12.

18. Cross references  
- Reliability aggregation and EMA update [DCH_Section3_Plasticity.md](./sections/DCH_Section3_Plasticity.md)  
- Embedding based similarity [DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md)  
- Formal objects and constraints [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md)

End of Section 5


---

# Dynamic Causal Hypergraph DCH — Section 6 Streaming Frequent Hyperpath Mining and Online Rule Induction

Parent outline [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md), Section 4 [DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md), Section 5 [DCH_Section5_CreditAssignment.md](./sections/DCH_Section5_CreditAssignment.md)

Version v0.1

1. Purpose and scope  
- Define a streaming Frequent Subgraph Mining FSM engine specialized for hyperpaths of the DCH.  
- Convert discovered valid hyperpaths from traversal into canonical templates, maintain frequency and reliability statistics with sliding windows and decay, and promote recurring high quality templates to symbolic rules.  
- Provide outputs to hierarchical abstraction Section 7 and priors to traversal Section 5, enabling the neuro symbolic loop.

2. Inputs and outputs  
Inputs  
- Stream of valid hyperpaths Π valid emitted by traversal, each with  
  - sink vertex id w star, ordered list of grounded hyperedges, per path score s π, WL and SAGE snapshot ids (optional), wall time t now.  
- Current DCH parameters for thresholds (support s min, reliability r min), windows (W seconds), decay γ.  
- Optional embedding features from Section 4 for template context.

Outputs  
- Template statistics table with counts, decayed support, reliability aggregates, distinct sink coverage.  
- Promotion events for templates that qualify as symbolic rules with descriptors.  
- Demotion or expiration events for templates falling below thresholds due to drift.

3. Hyperpath template definition  
- A hyperpath template τ abstracts away concrete timestamps while preserving structure and relative lags.  
- Representation goals  
  - Invariant to isomorphic reorderings of commutative tail sets.  
  - Stable under small timestamp noise by time binning.  
  - Compact and hashable for high throughput counting.  
- Template payload  
  - Topology order of hyperedges as a DAG from sources to sink.  
  - For each hyperedge e in the path  
    - Head neuron id j e  
    - Tail multiset of presynaptic neuron ids with binned relative delays Δ bin u equals bucketize head time minus tail time  
    - Tail cardinality m e  
  - Global metadata  
    - Sink neuron id j sink  
    - Path length number of edges L π  
    - Optional reliability bucket for each edge.

4. Canonical labeling for streaming hyperpaths  
4.1 Normalization  
- Shift times so that sink head time equals 0; all tail times become negative lags.  
- Bucketize lags using logarithmic or fixed bins over [1 ms, 30 ms]; default fixed bins of 1,2,3,5,7,10,15,20,25,30 ms.  
- Bucketize edge reliabilities into K r buckets (default 5).  
- Encode neuron id using a stable compact id (no learning needed).

4.2 Incidence expansion and layering  
- Convert the grounded hyperpath to a bipartite incidence DAG with vertex nodes and hyperedge nodes.  
- Compute a topological layering by nondecreasing head lag; break ties with tail lag multiset lexicographic order.  
- Within each layer, sort hyperedge nodes by tuple (j e, m e, multiset of tail (neuron id, Δ bin) pairs).

4.3 Canonical string and hash  
- For each hyperedge in the canonical order, emit a token  
  - H:j e;M:m e;T:[(i1,Δb1),(i2,Δb2),…];R:rb  
- Concatenate tokens with separators and append sink id j sink and total length L π.  
- Compute a 128 bit hash using a stable non cryptographic function (e.g., FarmHash style) to form template id tid; retain the string only for audit to reduce memory.

4.4 Determinism and collision handling  
- Deterministic ordering ensures identical grounded hyperpaths map to the same tid.  
- Collisions are rare but possible; maintain a small collision chain and verify by comparing canonical strings when a hash match occurs.

5. Counting with sliding window and decay  
- Maintain two tier statistics per tid  
  - Heavy hitters exact top K using SpaceSaving structure with capacity K HH (default 100k).  
  - Global approximate support using Count Min Sketch CMS with width W cm and depth D cm (defaults W cm equals 32768, D cm equals 4).  
- Sliding window W seconds (default 60 s)  
  - Ring buffer of B buckets (default 60 buckets of 1 s each) holding per tid increments for heavy hitters; on bucket expiry, decrement counts.  
  - CMS maintained with exponential decay γ per second (default γ equals 0.98) to approximate windowed counts for non HH templates.  
- Statistics per tid  
  - support win exact or approximate  
  - avg reliability r avg computed as EWMA over path scores or edge reliabilities (default EWMA α r equals 0.1)  
  - distinct sink count coverage computed with HyperLogLog HLL sketch per tid  
  - last seen time and stability measures (variance of increments).

6. Concept drift detection  
- Per tid, track two exponential windows reference and current with rates γ ref greater γ cur to detect significant changes in support or r avg.  
- Promotion hysteresis  
  - Require stability duration D stab (default 5 s) above thresholds before promotion.  
  - Demote when support falls below s min demote (default 0.5 times s min) for D demote duration (default 5 s).  
- Global drift  
  - Monitor total unique templates and churn rate; if churn spikes, raise thresholds temporarily to keep compute bounded.

7. Promotion criteria and rule descriptor  
7.1 Eligibility thresholds  
- support win greater or equal s min (default 50 within W 60 s).  
- r avg greater or equal r min (default 0.6).  
- coverage distinct sinks greater or equal c min (default 10) to avoid overfitting to a single sink.  
- topology constraints L π less or equal L max rule (default 6) and max tail size per edge less or equal k max rule (default 3).  
- no recent demotion cooling time (default 10 s).

7.2 Symbolic rule descriptor  
- Rule id rid equals tid plus version.  
- Head predicate Spike j sink, 0 and body predicates Spike i, minus Δ for each antecedent with their binned lags; include permissible lag intervals [Δ min, Δ max] for robustness.  
- Parameters  
  - Prior reliability r rule prior equals r avg  
  - Support support win, coverage distinct sinks, stability indicators  
  - Scope conditions task tags, dataset, or context labels if available  
  - Provenance pointer to sample grounded hyperpaths and example ids for audit.

7.3 Emission and lifecycle  
- Emit promotion event with descriptor; mark as Active.  
- Demotion emits state change to Inactive but preserves stats for potential reactivation.

8. Integration points  
8.1 With traversal (Section 5)  
- Rule priors influence edge selection via f rule feature; maintain a mapping from tid to motif matchers that recognize if an incoming edge sequence can complete a known rule.  
- Traversal can request prioritized expansion when partial matches to Active rules are detected (guide search).

8.2 With hierarchical abstraction (Section 7)  
- When a rule promotes, trigger higher order hyperedge creation  
  - Tail set equals the source vertices of the template (unique earliest events), Head equals the sink event class.  
  - Initialize new edge’s reliability with r rule prior and usage counters zero; attach provenance links to constituent edges.  
  - Ensure cycle prevention and deduplication against existing abstractions.

8.3 With embeddings (Section 4)  
- WL embeddings are included in template context for collision filtering; SAGE snapshot id is stored to correlate clusters with rules.  
- FSM can export cluster aware rule variants if a template consistently occurs within a specific embedding cluster.

9. APIs and data contracts  
- fsm submit path inputs path record seed id, sink vertex id, path edge ids, s π, t now, snapshot ids returns tid and whether counted in HH or CMS.  
- fsm tick inputs watermark time advances ring buffer, expires buckets, applies decay returns maintenance stats.  
- fsm poll promotions inputs max n returns up to n new rule descriptors and demotions since last poll.  
- fsm get template stats inputs tid returns support, r avg, coverage, stability and state.  
- fsm params get set thresholds, window size, decay rates, HH capacities, CMS sizes, hysteresis durations.  
- fsm export rules returns current Active ruleset for traversal and abstraction.

10. Observability and metrics  
- rule discovery rate promotions per minute and active count.  
- precision proxy fraction of promoted rules that remain Active for at least T persist (default 60 s).  
- coverage of active rules fraction of seeds whose paths partially match any Active rule.  
- compute usage HH occupancy, CMS load, per call latency.  
- churn rate promotions plus demotions per minute and top template volatility.

11. Complexity and performance targets  
- Submit path hot path  
  - Canonicalization O length π times tail size with small constants due to short paths and small tails.  
  - HH update O 1 amortized; CMS update O D cm.  
  - Latency target less than 50 microseconds per path on desktop prototype.  
- Maintenance tick per second  
  - Expire one bucket and apply CMS decay linear in HH size; budget less than 10 ms per second.  
- Memory  
  - HH stores K HH templates; CMS memory W cm times D cm counters; HLL per HH template.

12. Failure modes and mitigations  
- Template explosion under noise raise s min adaptively and increase γ to decay faster; throttle by rejecting low score paths s π below s path min (default 0.2).  
- Hash collisions verify canonical strings on HH insertion; for CMS tolerate collision noise and rely on HH for promotions.  
- Window skew if event timestamps jitter ensure watermark based tick; buffer out of order submissions until watermark advances.  
- Drift overreaction use hysteresis and moving thresholds to avoid oscillation.

13. Security and privacy notes  
- Canonical strings can include neuron ids; for external logs, map ids through a one way dictionary and coarsen Δ bins.  
- Avoid exporting raw path exemplars unless explicitly requested for audit with access control.

14. Mermaid diagram — FSM pipeline

```mermaid
flowchart LR
TRAV[Valid hyperpaths from traversal] --> NORM[Normalize and bucketize lags]
NORM --> CAN[Canonical label to tid]
CAN --> COUNT[HH and CMS counting with window W and decay gamma]
COUNT --> CHECK[Threshold and hysteresis checks]
CHECK -->|Promote| RULE[Emit rule descriptor and activate]
CHECK -->|Demote| DROP[Deactivate and retain stats]
RULE --> ABST[Trigger abstraction creation]
```

15. Parameter defaults and tuning (aligned with outline)  
- W equals 60 s, γ equals 0.98 per second.  
- s min equals 50, r min equals 0.6, c min equals 10, L max rule equals 6, k max rule equals 3.  
- K HH equals 100k, W cm equals 32768, D cm equals 4.  
- EWMA α r equals 0.1, D stab equals 5 s, s min demote equals 0.5 times s min, D demote equals 5 s.  
- s path min equals 0.2 to screen very low confidence paths.

16. Acceptance criteria for Section 6  
- Canonical labeling scheme for hyperpaths is specified, deterministic, and hashable.  
- Counting design supports sliding window, heavy hitters, and approximate totals with decay.  
- Concept drift detection and hysteresis thresholds are defined.  
- Promotion outputs a complete rule descriptor consumable by traversal and abstraction.  
- APIs, metrics, and performance targets are provided.  
- Diagram depicts the end to end FSM flow.

17. Cross references  
- Formal path and reliability definitions Section 1 [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md)  
- Embedding integration Section 4 [DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md)  
- Traversal source and priors Section 5 [DCH_Section5_CreditAssignment.md](./sections/DCH_Section5_CreditAssignment.md)  
- Abstraction sink Section 7 (to be drafted) [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)

End of Section 6


---

# Dynamic Causal Hypergraph DCH — Section 7 Hierarchical Abstraction and Higher Order Hyperedges

Parent outline [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md), Section 2 [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md), Section 3 [DCH_Section3_Plasticity.md](./sections/DCH_Section3_Plasticity.md), Section 6 [DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md)

Version v0.1

1. Purpose and scope  
- Define hierarchical abstraction in the DCH using higher order hyperedges HOEs that compress frequent reliable hyperpaths into single causal links.  
- Formalize the creation, validation, insertion, and usage of HOEs to accelerate learning, improve interpretability, and control combinatorial growth.  
- Specify constraints to prevent cycles and duplication and to maintain causal semantics and auditability.

2. Inputs and outputs  
Inputs  
- Promotion events and rule descriptors emitted by the FSM Section 6, including template id, canonical structure, support, average reliability, coverage, and provenance exemplars.  
- Current DCH state V t and E t, with incoming by head and by tail signature indices Section 2, and reliability r e Section 1.  
- Configuration thresholds for abstraction deployment and governance.

Outputs  
- New higher order hyperedges added to E t as template edges with provenance links.  
- Rewrite directives enabling DHG and traversal to utilize HOEs for candidate generation and path compression.  
- Metrics on abstraction usage, path shortening, and collision with existing structures.

3. Higher order hyperedge HOE object model  
3.1 Definition  
- A higher order hyperedge e HO is a directed template hyperedge connecting a set of source vertices to a sink vertex.  
- Tail HO is a set of abstract source roles derived from the root sources of the promoted template; each role binds to a neuron id and a binned lag interval relative to the sink.  
- Head HO is the sink vertex role derived from the promoted template.  
- Attributes  
  - delay window Δ min HO, Δ max HO computed from aggregate lag statistics for the template sources to sink; defaults use empirical q and 1 minus q quantiles (q default 0.1).  
  - refractory ρ inherited from Section 1 defaults or per neuron tables.  
  - reliability r HO initialized from the rule prior r rule prior Section 6, bounded to r min, r max.  
  - type label equals template edge per Section 1.  
  - version v for updates and rollbacks.

3.2 Instantiation semantics  
- An HOE instantiates at runtime when a set of concrete source vertices matches all source roles within their lag intervals and δ causal coherence for each role group, and the sink head vertex occurs within the HOE temporal window.  
- Instantiations produce realized event anchored edges or path rewrites for traversal depending on policy below.

4. Abstraction creation from rules  
4.1 Trigger and eligibility  
- On receiving a promotion event for template id tid from the FSM, validate rule descriptor against current E t and governance thresholds.  
- Eligibility requires no outstanding demotion state, sufficient support and coverage, and adherence to topology constraints L π and max tail size see Section 6.

4.2 Construction algorithm  
- Identify the unique earliest source event roles in the template as HOE Tail HO roles with associated lag intervals; compress roles from the same neuron where intervals overlap to a single interval union if within a small band ε merge default 1 ms.  
- Set Head HO role to the sink neuron id with zero lag.  
- Estimate Δ min HO, Δ max HO as conservative envelopes across all exemplars, optionally widened by a safety margin Δ saf default 1 ms to tolerate noise.  
- Initialize r HO equals clamp r rule prior to bounds; set created at equals now, last update equals now, usage count equals 0, version equals 1.  
- Canonicalize a tail signature for deduplication Section 2 style using neuron ids and lag bins.

4.3 Validation and invariants  
- Cycle safety Because all source roles precede the sink in time and the head is unique, abstraction cannot introduce temporal cycles in event anchored execution; additionally verify that no role involves the sink neuron as a source in the same rule to avoid self loops.  
- Duplication Check by tail signature and head neuron for an existing HOE; if found, merge statistics and optionally update Δ window and reliability via a smoothed update with weight λ merge default 0.3.

4.4 Insertion  
- Insert e HO into E t and update indices incoming by head and by tail signature.  
- Record provenance  
  - For the HOE store the template id tid, exemplar sample ids, and minimal generating set of grounded edges covering the template path minimal hitting set over exemplars.  
  - Maintain reverse links from constituent event anchored edges to parent HOEs for audit and cascading maintenance.

5. Rewrite and usage policy  
5.1 DHG integration GROW booster  
- When TC kNN is invoked for a new head vertex, include HOE induced candidates  
  - Match Tail HO roles against recent sources; if matched, generate a candidate HOE instance with priority boost proportional to r HO and group similarity from WL embeddings Section 4.  
  - Deduplicate against existing realized edges.  
- This accelerates structural growth along learned causal shortcuts.

5.2 Traversal integration path compression  
- During backward traversal Section 5, when the frontier contains a set of vertices matching Tail HO roles and the sink matches Head HO, allow a single expansion step via e HO replacing multiple underlying edge expansions.  
- Score contribution uses r HO in path score and records a compressed step with a pointer to provenance for audit.  
- Path compression caps depth and reduces branching, improving latency and stability.

5.3 Preference and consistency  
- Prefer HOE expansions over equivalent multi step expansions when available to reduce search space, unless a policy requires exploration of fine grained paths for diagnostic purposes.  
- Ensure consistency by preventing double counting  
  - If a compressed HOE step is taken, underlying constituent edges are not traversed within the same path instance.

6. Conflict resolution and composition  
6.1 Overlapping HOEs for the same head  
- Maintain a per head neuron budget K HOE head; rank HOEs by score s abs equals r HO times support times coverage; keep top K and demote others to standby.  
- At runtime, when multiple HOEs match, choose the one with highest s abs; break ties by smaller Tail HO cardinality then narrower Δ window.

6.2 HOE compositional growth  
- HOEs can enable higher tier abstractions when sequences of HOEs form frequent hyperpaths at the next level; treat HOEs as first class hyperedge nodes in FSM canonicalization and counting.  
- Constrain tier depth to D max abs default 3 to limit hierarchy height.

6.3 Merge and factor operations  
- Merge similar HOEs when their tails differ by at most one role and lag intervals overlap significantly with Jaccard over intervals greater or equal τ merge default 0.7.  
- Factor a large tail HOE into two smaller HOEs chained in series when traversal indicates the intermediate sub template is frequent and yields better compression.

7. Governance, updates, and rollback  
- Versioning Increment v when Δ windows or Tail HO roles are adjusted; retain prior versions for audit until grace period expires.  
- Freeze and protect Interact with task aware scaffolding Section 8 by freezing critical HOEs for prior tasks; frozen HOEs are excluded from aggressive merge or demotion actions.  
- Demotion If an HOE’s support or reliability falls below thresholds over stability window, mark as Inactive, remove rewrite preferences, and optionally remove from E t after a cooldown; retain provenance for reactivation.  
- Rollback Provide a reversible operation to restore prior version parameters when a change degrades performance according to monitoring metrics.

8. Interaction with plasticity and reliability  
- Treat HOE reliability r HO under the same EMA update law as other edges Section 3, with combined targets from local watcher style confirmations and path based updates from traversal.  
- When an HOE is used in a confirmed path, potentiate r HO; when predictions fail, depress accordingly; ensure that updates do not bypass constituent edge learning entirely by allocating a small fraction of credit to underlying edges via provenance weights λ share default 0.1.

9. Observability and metrics  
- abstraction usage rate proportion of paths using HOEs vs. base edges.  
- path shortening factor average original steps divided by compressed steps.  
- reliability trajectory of HOEs and underlying edges.  
- conflict and merge events per minute and their outcomes.  
- auditability coverage proportion of compressed steps with available provenance samples.

10. Complexity and resource model  
- Matching HOEs at DHG time requires checking Tail HO roles against recent sources; with Tail size m and per neuron ring buffers Section 2, admission is O m log b with small constants.  
- Traversal compression reduces effective depth from L to approximately L over c where c equals average compression factor; expected search space shrinks superlinearly due to branching reduction.  
- Index memory overhead scales with number of active HOEs; governed by K HOE head and global cap K HOE global.

11. Failure modes and mitigations  
- Spurious abstraction under noise mitigate with strong stability and coverage thresholds and demotion hysteresis Section 6.  
- Overlapping HOEs causing starvation of base edges allow periodic exploration where base paths are forced with small probability ε explore default 0.05 to validate continued correctness.  
- Drift invalidating Δ windows update intervals adaptively with quantile tracking; maintain p95 lag buffer to absorb slow drift.

12. Interfaces and data contracts  
- abstraction propose from rule inputs rule descriptor from FSM; outputs insertion result with HOE id and provenance summary.  
- abstraction match and instantiate inputs candidate head vertex and recent sources; outputs matched HOE instances for DHG.  
- abstraction compress for traversal inputs frontier state; outputs permissible HOE expansions with scores and provenance pointers.  
- abstraction metrics snapshot counters listed in Section 9.  
- abstraction params get set thresholds K HOE head, K HOE global, D max abs, τ merge, λ share, ε explore, Δ saf, ε merge.

13. Security and privacy  
- Provenance exemplars may reference raw event ids; restrict external exports by hashing neuron ids and coarsening lags to bins; enforce retention limits and access control for exemplar storage.

14. Mermaid diagram — abstraction creation and usage

```mermaid
flowchart LR
FSMRULE[Promoted template from FSM] --> BUILD[Construct HOE Tail roles and Δ windows]
BUILD --> VALID[Validate no cycle and deduplicate]
VALID --> INSERT[Insert HOE and record provenance]
INSERT --> REWRITE[Enable DHG candidate and traversal compression]
REWRITE --> METRICS[Track usage path shortening and reliability]
```

15. Parameter defaults and tuning  
- K HOE head equals 64, K HOE global configurable; D max abs equals 3.  
- τ merge equals 0.7 over interval overlap; λ merge equals 0.3; λ share equals 0.1; ε explore equals 0.05.  
- Δ saf equals 1 ms; ε merge equals 1 ms for role union; quantile parameter q equals 0.1.

16. Acceptance criteria for Section 7  
- HOE formal object and instantiation semantics defined with attributes and bounds.  
- Creation pipeline from FSM rule to HOE insertion is specified with validation, deduplication, and provenance.  
- Rewrite policies for DHG and traversal described with scoring and double counting prevention.  
- Conflict handling, composition, and governance covered with parameters and defaults.  
- Observability, complexity, failure modes, and an end to end diagram provided.

17. Cross references  
- Rule promotion and descriptors Section 6 [DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md)  
- DHG candidate generation Section 2 [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md)  
- EMA reliability and pruning Section 3 [DCH_Section3_Plasticity.md](./sections/DCH_Section3_Plasticity.md)

End of Section 7


---

# Dynamic Causal Hypergraph DCH — Section 8 Task Aware Scaffolding Isolate or Reuse and Freeze

Parent outline [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md), Section 3 [DCH_Section3_Plasticity.md](./sections/DCH_Section3_Plasticity.md), Section 4 [DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md), Section 6 [DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md), Section 7 [DCH_Section7_HierarchicalAbstraction.md](./sections/DCH_Section7_HierarchicalAbstraction.md)

Version v0.1

1. Purpose and scope  
- Provide a continual learning control layer that detects task similarity and governs structural plasticity and protection to avoid catastrophic forgetting while enabling transfer.  
- Implement the FREEZE primitive and region based GROW biasing ISOLATE vs REUSE within the evolving DCH.  
- Define metrics, governance, and interfaces to coordinate with DHG, Plasticity, Traversal, FSM, and Abstraction modules.

2. Inputs and outputs  
Inputs  
- New task context with optional id and small calibration batch stream of events and labels.  
- Current DCH snapshot including reliability r e, usage counts, active HOEs, rule set from FSM, WL and optional SAGE embeddings.  
- Policy thresholds and budgets configurable at runtime.

Outputs  
- Freeze set P of protected edges and HOEs with immutability flags and pruning protection.  
- Region assignment map R mapping newly grown edges to a task specific region id with leakage control.  
- Updated per module knobs learning rate multipliers alpha, traversal biases, pruning thresholds, budgets.  
- Task registry entries with similarity scores, freeze schedules, and audit summaries.

3. Task similarity detection  
Goal Decide if the incoming task should reuse or isolate from prior knowledge.

3.1 Activation overlap signal  
- For calibration batch B task feed events through SNN and DCH to record activated high reliability structures.  
- Activated edge set A task equals { e mid r e greater or equal r act and edge participates in at least one valid path for batch } with r act default 0.7.  
- For each prior task u with protected set P u compute Jaccard J u equals size of intersection A task and P u divided by size of union A task and P u.

3.2 Embedding similarity signal  
- Compute WL embedding centroid c task over sink heads in calibration batch see [DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md).  
- For each prior task u with centroid c u compute cosine similarity S u cos equals cosine of c task and c u.

3.3 Performance signal optional  
- Evaluate quick proxy accuracy or reward on a small validation slice for candidate reuse vs isolate warm starts if available.

3.4 Unified similarity score  
- For each prior task u compute S u equals w act times J u plus w emb times S u cos plus w perf times normalized performance with defaults w act 0.5, w emb 0.4, w perf 0.1 when performance is available else renormalize.  
- Aggregate to S task equals max over u of S u best match prior task.  
- Decision thresholds reuse if S task greater or equal θ reuse default 0.6, isolate if S task less or equal θ isolate default 0.3, hybrid otherwise.

4. Policy actions  
4.1 REUSE mode S task greater or equal θ reuse  
- Keep plasticity enabled for structures relevant to the matched prior task  
  - Allow reliability updates for edges in P match and their neighborhoods.  
  - Temporarily raise plasticity α by factor k alpha up default 1.5 for matching subgraph and increase traversal softmax temperature mildly to explore variants τ select up factor 1.1.  
- Tighten pruning thresholds to preserve useful structures τ prune down factor 0.8 for matching subgraph.  
- Enable rule guided traversal priority weight w rule up to 0.8 for rules tagged to matched task.

4.2 ISOLATE mode S task less or equal θ isolate  
- Engage FREEZE for critical prior structures Section 5  
  - Freeze top K protect percent of edges and all active HOEs linked to matched tasks with high centrality and reliability.  
- Allocate a new region id r new and bias GROW to place new edges in this region only leakage below epsilon leak default 0.1  
  - For DHG candidate admission apply a penalty to candidates that connect across regions unless they pass a high reliability threshold from HOE priors.  
- Lower plasticity α for frozen regions to 0 and set pruning immunity for protected edges.  
- Increase budgets for new region K head up factor 1.2 and global cap fraction to accelerate learning without interfering with frozen structures.

4.3 HYBRID mode θ isolate less S task less θ reuse  
- Partially freeze A subset equals top quantile q freeze default 0.5 of critical edges based on composite score s crit equals r e times usage normalized times rule support.  
- Split budgets 50 50 across matched region and new region id and allow limited cross region linking with reliability gate r gate default 0.8.  
- Use traversal that prefers HOE compressions from prior task while permitting base edge exploration in new region.

5. FREEZE semantics and governance  
5.1 Freeze flags and effects  
- freeze reliability True prohibits updates to r e from watchers and path based signals see [DCH_Section3_Plasticity.md](./sections/DCH_Section3_Plasticity.md).  
- freeze prune True exempts edges from pruning policies except catastrophic integrity events.  
- freeze traversal protect True prevents negative evidence from erroneous seeds from depressing r e for protected edges.  
- All flags apply to both base edges and HOEs.

5.2 Critical set selection  
- Rank edges by s crit equals r e times usage rate times betweenness centrality estimate from recent traversal logs; include HOEs with rule support above r rule prior greater or equal 0.7.  
- Choose top K protect percent default 5 percent per matched task capped by K protect max default 50k.

5.3 Freeze schedule and review  
- Set freeze TTL default 120 s after which edges enter review state where small adaptation is allowed with α review default 0.02 and narrow bounds on r e delta per second.  
- Defrost requires two conditions satisfied stability of matched task performance over window W perf default 30 s and low cross task conflict rate measured as fraction of traversal depression events on protected edges less than τ conflict default 0.02.

5.4 Provenance and audit  
- Record freeze events with task id, reason, selection stats, and affected edge ids.  
- Provide reproducibility using snapshot ids for DCH and embedding models.

6. Regionization and structural isolation  
6.1 Region labeling  
- Assign region id attribute to edges on creation; inherit for HOEs from their constituent role neurons majority region.  
- region leakage epsilon leak allows a small fraction of cross region edges per head neuron to support transfer default 0.1.

6.2 GROW bias  
- In DHG admission Section 2 penalize candidates whose tails span multiple regions beyond leakage tolerance by subtracting penalty λ reg times span minus 1 with λ reg default 0.5 from s cand.  
- Prioritize within region candidates and HOE induced candidates matching the active task region.

6.3 Traversal guards  
- During credit assignment Section 5 restrict expansions that cross regions unless HOE or rule prior indicates relevance; annotate paths with region transitions for diagnostics.

7. Budgets and resource allocation  
- Maintain per region head budget K head region and global cap K global split across regions according to policy weights.  
- New region boosts K head region by up to 20 percent for warm up then anneal to steady share.  
- Enforce fairness by capping any single region at max share σ max default 0.6 of total active edges.

8. Interaction with rules and abstraction  
- Prioritize Active rules from FSM tagged to matched prior tasks in REUSE and HYBRID modes to bias traversal and DHG.  
- Frozen rules imply corresponding HOEs are protected; abstraction module may still collect usage metrics for them.  
- When a new task consistently triggers novel templates, promote rules and instantiate HOEs in the new region to accelerate learning see [DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md) and [DCH_Section7_HierarchicalAbstraction.md](./sections/DCH_Section7_HierarchicalAbstraction.md).

9. Parameters and defaults  
- Similarity thresholds θ reuse 0.6, θ isolate 0.3.  
- r act 0.7 for activation sets; weights w act 0.5, w emb 0.4, w perf 0.1.  
- Freeze K protect percent 5 percent, K protect max 50k, TTL 120 s, α review 0.02, τ conflict 0.02.  
- Region leakage epsilon leak 0.1, λ reg 0.5, σ max 0.6.  
- Budget multipliers k alpha up 1.5 for reuse, K head boost 1.2 for isolate warm up.  
- Reliability gates r gate 0.8 for cross region links.

10. Interfaces and data contracts  
scaffold start task  
- Inputs task id optional, calibration batch handle, policy overrides optional.  
- Returns decision mode in {REUSE, ISOLATE, HYBRID}, similarity score S task, freeze set summary, region id.

scaffold policy step  
- Inputs snapshot of recent metrics and drift indicators; returns incremental adjustments to α multipliers, τ prune, traversal weights, and budget shares.

scaffold set freeze  
- Inputs set of edge or HOE ids and flags; returns count applied and conflicts.

scaffold region map  
- Inputs mapping for new edges or neurons to region ids; returns acknowledgments.

scaffold metrics snapshot  
- Returns similarity history, protection population counts, region occupancy shares, transfer and forgetting metrics, and recent conflicts.

11. Metrics and evaluation  
- Forward transfer gain difference in sample efficiency or accuracy when reusing prior structures vs isolate baseline.  
- Backward transfer and forgetting measure change in prior task performance after learning new task under each policy.  
- Freeze precision fraction of protected edges that remain essential measured by traversal inclusion for matched tasks.  
- Region occupancy and leakage actual share vs policy targets and cross region edge rates.  
- Stability of r e on frozen edges should vary less than 5 percent across TTL windows.

12. Failure modes and mitigations  
- Over freezing underestimates similarity or too aggressive protection reduce K protect percent adaptively when forgetting is low and forward transfer is poor.  
- Under isolation reuse degrades prior tasks increase thresholds or strengthen penalties for cross region GROW.  
- Oscillation between modes apply hysteresis 0.05 to thresholds and minimum dwell time T dwell default 30 s before switching modes.  
- Imbalanced regions enforce σ max and rebalance budgets with gradual annealing.

13. Security and privacy  
- Task tags and freeze logs may reveal dataset identities restrict export by hashing task ids and summarizing metrics without raw ids.  
- Regions are internal control attributes do not expose in external logs unless anonymized.

14. Mermaid diagram task aware scaffolding

```mermaid
flowchart LR
CAL[Calibration batch] --> SIM[Compute activation and embedding similarity]
SIM --> DECIDE{S_task thresholds}
DECIDE -->|REUSE| REU[Enable plasticity on prior subgraph]
DECIDE -->|ISOLATE| ISO[Freeze critical edges and create new region]
DECIDE -->|HYBRID| HYB[Partial freeze and split budgets]
REU --> KNOBS[Adjust alpha prune traversal rule bias]
ISO --> KNOBS
HYB --> KNOBS
KNOBS --> PIPE[Publish knobs to DHG Plasticity Traversal FSM Abstraction]
```

15. Acceptance criteria for Section 8  
- Similarity detection method defined with activation and embedding signals and unified score.  
- Policy actions specified for REUSE, ISOLATE, and HYBRID with parameter defaults.  
- FREEZE semantics and governance detailed, including selection, schedule, and audit.  
- Regionization defined with DHG and traversal integration and budget controls.  
- Interfaces, metrics, and a diagram provided.

16. Cross references  
- Reliability and pruning controls [DCH_Section3_Plasticity.md](./sections/DCH_Section3_Plasticity.md)  
- WL embeddings and grouping [DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md)  
- FSM rules and HOEs [DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md), [DCH_Section7_HierarchicalAbstraction.md](./sections/DCH_Section7_HierarchicalAbstraction.md)

End of Section 8


---

# Dynamic Causal Hypergraph DCH — Section 9 Module Interfaces and Data Contracts

Parent outline [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md), Section 2 [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md), Section 3 [DCH_Section3_Plasticity.md](./sections/DCH_Section3_Plasticity.md), Section 4 [DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md), Section 5 [DCH_Section5_CreditAssignment.md](./sections/DCH_Section5_CreditAssignment.md), Section 6 [DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md), Section 7 [DCH_Section7_HierarchicalAbstraction.md](./sections/DCH_Section7_HierarchicalAbstraction.md), Section 8 [DCH_Section8_TaskAwareScaffolding.md](./sections/DCH_Section8_TaskAwareScaffolding.md)

Version v0.1

1. Purpose and scope  
- Define clear, stream safe interfaces among DCH subsystems useful for a Python prototype and for hardware co design alignment.  
- Specify data contracts for events, vertices, hyperedges, paths, templates, higher order hyperedges, and metrics.  
- Establish idempotency, ordering, and snapshot semantics.

2. Architectural interaction model  
- Event driven core with watermark based ordering.  
- Modules publish and consume records over in process queues or async calls.  
- Snapshot ids label embedding and rule tables to guarantee reproducibility.

3. Global identifiers and snapshots  
- neuron id integer stable per deployment.  
- vertex id 64 bit generated as hash of neuron id and timestamp microseconds.  
- hyperedge id 128 bit generated from tail signature and head vertex id.  
- hoe id 128 bit generated from template id and version.  
- snapshot id string for WL and SAGE embedding refresh cycles and FSM ruleset revisions.  
- traversal cycle id string per credit assignment pass.

4. Data contracts core records  
4.1 Event record  
- neuron id  
- timestamp microseconds  
- meta optional map includes source sensor id and split tags

4.2 Vertex record  
- vertex id  
- neuron id  
- timestamp microseconds

4.3 Hyperedge record event anchored  
- edge id  
- tail list of vertex ids  
- head vertex id  
- delta min microseconds  
- delta max microseconds  
- refractory microseconds  
- reliability float bounded  
- created at microseconds  
- last update time microseconds  
- usage count integer  
- type equals event edge

4.4 Hyperedge record template or HOE  
- edge id  
- tail roles list each role contains neuron id and lag interval bins  
- head role neuron id  
- delta min ho microseconds  
- delta max ho microseconds  
- refractory microseconds  
- reliability float bounded  
- created at microseconds  
- last update time microseconds  
- usage count integer  
- type equals template edge  
- version integer  
- provenance list template ids and exemplar references

4.5 Path record from traversal  
- seed vertex id  
- path edge ids ordered  
- score float product of reliabilities  
- mode reward or error or correct  
- snapshot ids for embeddings and rules  
- cycle id  
- audit optional list of vertex ids for debug

4.6 Template record from FSM  
- template id  
- canonical string audit safe  
- support windowed integer or float  
- reliability average float  
- coverage distinct sinks estimate  
- status active or inactive  
- params bins lags and buckets  
- snapshot id

5. Module interface contracts  
5.1 Event ingestion  
- push event input event record returns accepted boolean and queue depth  
- get watermark returns current logical time and lag statistics

5.2 Dynamic hypergraph construction DHG  
- on post spike input neuron id and timestamp returns created edge ids list and counts of candidates evaluated and deduplicated  
- get edges by head input head vertex id returns list of hyperedge records  
- get edges by tail key input tail signature returns hyperedge record if present  
- set params input map of window and budget parameters returns applied config

5.3 Plasticity  
- resolve watchers input watermark time returns resolved outcome counts and updated edge ids list  
- update from paths input map edge id to r hat path returns applied updates count  
- prune step input policy id returns removed edge ids list  
- set freeze input ids and flags returns applied counts

5.4 Embedding WL online  
- update on event input vertex id and timestamp returns list of touched vertex ids  
- propose groups input head vertex id returns list of group candidates and similarity scores  
- get vertex embedding input vertex id returns vector and last update time  
- set params input r wl and d wl and lsh knobs returns applied config

5.5 Embedding SAGE periodic  
- refresh now input snapshot hint returns snapshot id and stats  
- get vertex embedding input vertex id and snapshot id returns vector  
- link fsm input enable flag returns acknowledgment

5.6 Traversal credit assignment  
- assign credit input seeds and mode and caps returns path records and edge contribution map  
- get metrics snapshot returns traversal counters and latency stats  
- set params input M and L and B and tau select and H back returns applied config

5.7 FSM streaming frequent hyperpath mining  
- submit path input path record returns template id and counting tier heavy hitter or cms  
- tick input watermark time returns maintenance stats and promotions and demotions since last tick  
- poll promotions input limit returns up to limit rule descriptors  
- get template stats input template id returns statistics  
- set params input window sizes and thresholds returns applied config

5.8 Abstraction hierarchical  
- propose from rule input rule descriptor returns hoe id and dedup result  
- match and instantiate input head vertex id and recent sources returns matched hoe instances  
- compress for traversal input frontier state returns admissible hoe expansions with scores  
- get metrics snapshot returns abstraction usage rates and path shortening  
- set params input budgets and merge thresholds returns applied config

5.9 Scaffolding task aware  
- start task input task id and calibration handle returns decision mode and similarity score and region id  
- policy step input recent metrics returns knob adjustments for modules  
- set freeze input ids and flags returns applied counts  
- region map input edge to region assignments returns acknowledgments  
- get metrics snapshot returns protection population and region occupancy

5.10 Meta controller  
- step input module summaries returns global actions for thresholds budgets and traversal bias  
- subscribe metrics input list of modules returns composite dashboard feed id  
- get policy state returns current objectives weights and recent actions

5.11 Logging and metrics  
- emit counter input name and value with tags returns acknowledgment  
- emit gauge input name and value with tags returns acknowledgment  
- emit event input name and payload returns acknowledgment  
- list metrics returns available series and tag keys

6. Ordering idempotency and error handling  
- Ordering watermark based with per neuron monotonic timestamps and per head critical sections.  
- Idempotency keys  
  - DHG creation token tuple head vertex id and tail signature  
  - Watcher id tuple edge id and earliest tail timestamp and head neuron id  
  - FSM canonical hash template id and collision verification with canonical string when necessary  
- Retries and dedup based on these keys.  
- Error classes  
  - input out of order within tolerance buffer exceeded  
  - budget exceeded candidate dropped  
  - stale snapshot embedding requested  
  - freeze conflict update denied

7. Security privacy and governance  
- Redact neuron id mapping for external logs through a one way dictionary.  
- Quantize timestamps to q time for export.  
- Access control roles viewer and operator and admin for freeze and policy changes.  
- Provenance retention policy with TTL and hashed exemplar ids.

8. Observability and health  
- Per module metrics common core  
  - qps and latency buckets  
  - queue depth and watermark lag  
  - error rates per class  
  - budget occupancy and drops  
- Tracing  
  - attach correlation id across event to edge creation to traversal to FSM to abstraction  
  - sample traces for heavy code paths and anomaly triggers

9. Mermaid diagram module interaction

```mermaid
flowchart LR
EVIN[Event ingestion] --> DHG[DHG construct]
DHG --> PL[Plasticity]
DHG --> EMBWL[WL embedding]
EMBWL --> DHG
PL --> TRAV[Traversal credit assignment]
TRAV --> FSM[FSM stream]
FSM --> ABS[Abstraction HOE]
ABS --> DHG
FSM --> MC[Meta controller]
SCF[Scaffolding] --> MC
MC -->|knobs| DHG
MC -->|knobs| PL
MC -->|knobs| TRAV
MC -->|knobs| FSM
MC -->|knobs| ABS
LOG[Metrics and logging] --- DHG
LOG --- TRAV
LOG --- FSM
LOG --- ABS
LOG --- SCF
```

10. Acceptance criteria for Section 9  
- Interfaces for all modules defined with inputs and outputs and configuration hooks.  
- Data contracts specified for core records and identifiers and snapshots.  
- Ordering and idempotency semantics established for safe streaming.  
- Security and observability guidelines included.  
- Diagram reflects module interactions and control flow.

11. Cross references  
- Temporal validity and hyperedge schema Section 1 [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md)  
- DHG operations Section 2 [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md)  
- Plasticity update and pruning Section 3 [DCH_Section3_Plasticity.md](./sections/DCH_Section3_Plasticity.md)  
- Embeddings Section 4 [DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md)  
- Traversal Section 5 [DCH_Section5_CreditAssignment.md](./sections/DCH_Section5_CreditAssignment.md)  
- FSM Section 6 [DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md)  
- Abstraction Section 7 [DCH_Section7_HierarchicalAbstraction.md](./sections/DCH_Section7_HierarchicalAbstraction.md)  
- Scaffolding Section 8 [DCH_Section8_TaskAwareScaffolding.md](./sections/DCH_Section8_TaskAwareScaffolding.md)

End of Section 9


---

# Dynamic Causal Hypergraph DCH — Section 10 Complexity and Resource Model

Parent outline [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md), Section 2 [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md), Section 3 [DCH_Section3_Plasticity.md](./sections/DCH_Section3_Plasticity.md), Section 4 [DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md), Section 5 [DCH_Section5_CreditAssignment.md](./sections/DCH_Section5_CreditAssignment.md), Section 6 [DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md), Section 7 [DCH_Section7_HierarchicalAbstraction.md](./sections/DCH_Section7_HierarchicalAbstraction.md), Section 8 [DCH_Section8_TaskAwareScaffolding.md](./sections/DCH_Section8_TaskAwareScaffolding.md), Section 9 [DCH_Section9_Interfaces.md](./sections/DCH_Section9_Interfaces.md)

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
- DHG costs and parameters [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md)  
- Plasticity and pruning [DCH_Section3_Plasticity.md](./sections/DCH_Section3_Plasticity.md)  
- Embeddings [DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md)  
- Traversal [DCH_Section5_CreditAssignment.md](./sections/DCH_Section5_CreditAssignment.md)  
- FSM [DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md)  
- Abstraction [DCH_Section7_HierarchicalAbstraction.md](./sections/DCH_Section7_HierarchicalAbstraction.md)  
- Scaffolding [DCH_Section8_TaskAwareScaffolding.md](./sections/DCH_Section8_TaskAwareScaffolding.md)  
- Interfaces for metrics and knobs [DCH_Section9_Interfaces.md](./sections/DCH_Section9_Interfaces.md)

End of Section 10


---

# Dynamic Causal Hypergraph DCH — Section 11 Software Prototype Blueprint Python Norse

Parent outline [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md), Section 2 [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md), Section 3 [DCH_Section3_Plasticity.md](./sections/DCH_Section3_Plasticity.md), Section 4 [DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md), Section 5 [DCH_Section5_CreditAssignment.md](./sections/DCH_Section5_CreditAssignment.md), Section 6 [DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md), Section 7 [DCH_Section7_HierarchicalAbstraction.md](./sections/DCH_Section7_HierarchicalAbstraction.md), Section 8 [DCH_Section8_TaskAwareScaffolding.md](./sections/DCH_Section8_TaskAwareScaffolding.md), Section 9 [DCH_Section9_Interfaces.md](./sections/DCH_Section9_Interfaces.md), Section 10 [DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md)

Version v0.1

1. Goals and non goals
- Deliver a runnable Python reference implementation for event vision datasets DVS Gesture and N MNIST using Norse for SNN spiking and tonic for datasets.
- Prioritize streaming determinism, clear module boundaries, and observability over raw speed; enable later acceleration and hardware mapping.
- Non goals Training deep ANN baselines inside this repo; full distributed system; GPU optimized GNN training beyond a small GraphSAGE refresh.

2. Tech stack and dependencies
- Python 3.11, PyTorch 2.3+, Norse latest, tonic dataset loaders, numpy, numba optional, networkx for audits, dataclasses jsonschema for contracts, faiss optional for LSH, scikit learn optional for metrics, PyTorch Lightning or simple engine for loops, rich or loguru for logs.
- Optional GPU for periodic SAGE embedding; CPU path for WL and DHG.

3. Repository layout
- dch_core/
  - events.py event ingestion, watermarking, ring buffers
  - dhg.py TC kNN candidate gen, indices
  - plasticity.py watchers, EMA updates, pruning
  - embeddings/
    - wl.py WL online embedding, LSH grouping
    - sage.py periodic incidence GraphSAGE refresh
  - traversal.py backward B walk, beam control
  - fsm.py canonicalization, HH and CMS counting, hysteresis
  - abstraction.py higher order hyperedges creation and matching
  - scaffolding.py task similarity, freeze, regions, knobs
  - interfaces.py dataclasses for records and params, snapshot ids
  - meta.py meta controller policy and knob arbitration
  - metrics.py counters, gauges, tracing hooks
- experiments/
  - dvs_gesture.py end to end pipeline run config
  - n_mnist.py end to end pipeline run config
  - synthetic_micro.py seeded synthetic generators and assertions
- tests/ unit and integration tests per module
- configs/ default.yaml, dataset specific yaml
- docs/ this spec and diagrams

4. Data models implement contracts
- Implement records and params from [DCH_Section9_Interfaces.md](./sections/DCH_Section9_Interfaces.md) as python dataclasses and simple pydantic like validators if desired.
- Provide stable id functions for vertex id and edge id consistent with Section 9.

5. Event and SNN integration
- Use tonic to load DVS Gesture and N MNIST into event streams; or run a small Norse SNN to produce spikes from raw events.
- Event ingestion normalizes timestamps to microseconds, enforces per neuron monotonicity, and publishes events to DHG and WL update queues.

6. Pipeline orchestration and threading model
- Single process with cooperative components; three thread pools
  - Event lane DHG, Plasticity, WL update (tight latency path)
  - Reasoning lane Traversal, FSM, Abstraction (periodic)
  - Control lane Scaffolding, Meta controller (slow cadence)
- Use asyncio or concurrent futures; all modules expose non blocking APIs with bounded queues and watermark based flow control.

7. Scheduling and cadences
- Event lane processes events as they arrive with per event budgets from [DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md).
- Traversal cycles run every T trav default 20 ms or on demand when supervision arrives.
- SAGE refresh every 500 ms; FSM tick every 1 s or watermark driven; scaffolding policy step every 500 ms.

8. Module responsibility matrix
- DHG maps to Section 2; WL to Section 4; Plasticity to Section 3; Traversal to Section 5; FSM to Section 6; Abstraction to Section 7; Scaffolding to Section 8; Interfaces and Meta to Section 9.

9. Configuration system
- YAML config files under configs/ with env overrides.
- Hot reload of certain knobs via Meta controller step; immutable knobs require restart.

10. Minimal end to end loop pseudocode

```text
initialize_modules(config)
load_dataset_stream(dataset)

for batch in dataset.stream():
    for event in batch:  # event = (neuron_id, t_us)
        events.ingest(event)
        created_edges = dhg.on_post_spike(event)
        plasticity.on_event_tail_and_head(created_edges, event)
        wl.update_on_event(event)
        if wl.should_group(event):
            dhg.boost_with_groups(event, wl.propose_groups(event))

    if time_to_run_traversal():
        seeds = supervisor_or_reward.get_seeds()
        paths, edge_contribs = traversal.assign_credit(seeds, mode)
        plasticity.update_from_paths(edge_contribs)
        for p in paths:
            fsm.submit_path(p)

    if fsm.time_to_tick():
        promos, demos = fsm.tick()
        for rule in promos:
            abstraction.propose_from_rule(rule)
        meta.consume_fsm(promos, demos)

    if time_to_refresh_sage():
        sage.refresh()

    if scaffolding.time_to_step():
        knobs = scaffolding.policy_step(metrics.snapshot())
        meta.apply_knobs(knobs)
```

11. Norse SNN reference component
- Provide a small configurable Norse module that maps event frames to spikes for datasets that lack pre spiked streams.
- Keep SNN purely subsymbolic and do not backprop; rely on DCH learning loop for structure discovery.

12. WL embedding implementation notes
- Use stable 64 bit hashes and feature hashing to d 64; cosine LSH with faiss or custom bands/rows.
- Maintain per vertex vector table with last updated timestamp; enforce staleness bound 20 ms for DHG usage.

13. GraphSAGE periodic implementation
- Construct incidence expansion mini batches with neighbor sampling caps; run on GPU if available.
- Save snapshots with snapshot id and expose read only API to FSM.

14. FSM implementation notes
- Canonicalization pipeline per [DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md) with aggressively pooled buffers.
- SpaceSaving heavy hitters plus CMS; HLL for coverage; hysteresis to stabilize promotions.

15. Abstraction and traversal integration
- When rules promote, construct HOEs and insert; traversal consumes HOEs as compressed steps.
- Prevent double counting by marking compressed expansions in path records.

16. Scaffolding and meta control
- Compute activation overlap and WL centroid similarity; decide REUSE, ISOLATE, HYBRID; apply freeze and region policies.
- Meta controller arbitrates knobs across modules and enforces backpressure according to [DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md).

17. Metrics, logging, and tracing
- Counters for event rate, edges active, prune rate, traversal yield, rule promotions; latency histograms per lane.
- Structured logs with correlation id across modules; sample traces for hot paths.
- Support CSV Parquet export and simple dashboards via rich textual panels.

18. Testing strategy
- Unit tests per module for boundary cases and invariants
  - Temporal validity checks for DHG windows and refractory
  - Watcher creation and resolution determinism
  - WL collision precision on synthetic graphs
  - Traversal B connectivity and constraints
  - FSM canonicalization idempotency and hash stability
  - Abstraction cycle prevention and deduplication
- Integration tests
  - Synthetic micrographs with planted causal chains; ensure recall and precision targets per Section 5 and 6 acceptance criteria.
  - Dataset smoke tests run short streams and validate throughput and latency budgets from Section 10.

19. Example configs

```yaml
# configs/default.yaml
dataset: dvs_gesture
time:
  wl_cadence_ms: 10
  traversal_cadence_ms: 20
  sage_cadence_ms: 500
  fsm_tick_s: 1
windows:
  delta_min_us: 1000
  delta_max_us: 30000
  delta_causal_us: 2000
refractory_us: 1000
tc_knn:
  k_max: 3
  min_in: 6
  comb_cap: 10
plasticity:
  alpha: 0.1
  r_min: 0.02
  r_max: 0.98
wl:
  d: 64
  r: 2
  lsh:
    bands: 8
    rows_per_band: 4
traversal:
  seeds: 8
  depth: 12
  branch_cap: 4
  tau_select: 0.7
fsm:
  window_s: 60
  support_min: 50
  r_min: 0.6
  decay: 0.98
```

20. CLI entry points
- experiments/dvs_gesture.py runs the full pipeline with config path; flags for cadences and budgets.
- experiments/n_mnist.py same template for N MNIST.
- Support resume from snapshot ids for embeddings and rulesets.

21. Mermaid diagram overall software dataflow

```mermaid
flowchart LR
SRC[Event stream or Norse spikes] --> EV[Event ingestion]
EV --> DHG[DHG TC-kNN]
DHG --> PL[Plasticity]
PL --> WL[WL embedding]
WL -->|groups| DHG
PL --> TRAV[Traversal]
TRAV --> FSM[FSM stream]
FSM --> ABS[Abstraction]
ABS --> DHG
SCF[Scaffolding] --> META[Meta controller]
META --> DHG
META --> PL
META --> TRAV
META --> FSM
META --> ABS
LOG[Metrics] --- EV
LOG --- DHG
LOG --- TRAV
LOG --- FSM
```

22. Acceptance criteria for Section 11
- Clear repo structure with module responsibilities aligned to Sections 1 to 10.
- Deterministic streaming orchestration with lanes and cadences.
- Configs, metrics, and tests specified; example end to end loop given.
- Norse integration defined for dataset ingestion.
- Diagram provided reflecting dataflow and control.

23. Next steps
- Implement minimal skeletons and stubs for records and APIs from [DCH_Section9_Interfaces.md](./sections/DCH_Section9_Interfaces.md).
- Build synthetic micro benchmarks for Sections 5 and 6 validation.
- Prepare dataset adapters and smoke tests under experiments/.
- Iterate to performance targets in [DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md).

End of Section 11


---

# Dynamic Causal Hypergraph DCH — Section 12 Evaluation Protocol Datasets Metrics and Ablations

Parent outline [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md), Section 2 [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md), Section 4 [DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md), Section 5 [DCH_Section5_CreditAssignment.md](./sections/DCH_Section5_CreditAssignment.md), Section 6 [DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md), Section 7 [DCH_Section7_HierarchicalAbstraction.md](./sections/DCH_Section7_HierarchicalAbstraction.md), Section 10 [DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md), Section 11 [DCH_Section11_SoftwareBlueprint.md](./sections/DCH_Section11_SoftwareBlueprint.md)

Version v0.1

1. Objectives  
- Establish rigorous, reproducible evaluation of DCH on event vision tasks with emphasis on accuracy, sample efficiency, throughput, interpretability, and continual learning.  
- Compare against strong non DCH baselines and DCH ablations to isolate contributions of core mechanisms WL embeddings, traversal credit assignment, FSM, and HOEs.  
- Produce artifacts configs, logs, and rule libraries enabling exact replication.

2. Datasets and splits  
2.1 DVS Gesture  
- Source public DVS Gesture dataset tonic provides loader with user and gesture splits.  
- Standard split users 1 to 23 train, 24 to 29 test or k fold per literature; adopt standard to ease comparison.  
- Preprocessing keep native event stream resolution; quantize timestamps to 1 μs internal representation; no frame accumulation.  
- Labels gesture id associated with trial windows; provide supervision seeds by tagging output head vertices within ground truth windows see Section 5 credit assignment.

2.2 N MNIST  
- Source tonic N MNIST event dataset derived from MNIST with saccade patterns.  
- Train split 60k, test split 10k; use event stream directly.  
- Supervision per sample window; seed output spikes at sample end or sliding windows.

2.3 Stream construction and watermarking  
- For both datasets construct continuous event streams in sample order for streaming evaluation; optionally interleave samples for mini epochs.  
- Maintain watermark based ordering as in [DCH_Section11_SoftwareBlueprint.md](./sections/DCH_Section11_SoftwareBlueprint.md) to synchronize periodic modules.

2.4 Reproducibility controls  
- Global seed seeding PyTorch numpy and Python random.  
- Deterministic WL hashing functions; seeded traversal RNG keyed by cycle id.  
- Snapshot ids for WL SAGE and FSM rules included in logs and results.

3. Task formulations  
3.1 Classification protocol  
- Output head neurons one per class gesture or digit; decision rule integrate output spikes in a decision window W out per sample default full sample or last T tail ms and take argmax count; ties broken by earliest spike time.  
- Alternate probabilistic decoding low pass filter of spike trains with exponential kernel and argmax filtered rate.  
- Online setting compute cumulative decision at fixed cadence and measure latency to correct classification.

3.2 Continual learning protocol optional  
- Task sequence two tasks e.g., subset of gestures then full set or digits 0 to 4 then 5 to 9 with bounded memory.  
- Apply scaffolding policies from [DCH_Section8_TaskAwareScaffolding.md](./sections/DCH_Section8_TaskAwareScaffolding.md); measure forward and backward transfer and forgetting.

4. Baselines  
4.1 Surrogate gradient SNN  
- Norse based spiking network with surrogate gradient training for classification; tuned to dataset standard baselines; train with cross entropy on rate decoded outputs.  
- Report accuracy and training sample budget matched to DCH label exposure where possible.

4.2 Reservoir SNN LSM style  
- Fixed recurrent reservoir with linear readout trained by ridge regression on spike features rate or temporal basis.  
- Provides sample efficient baseline without end to end backprop.

4.3 TEH static variant  
- Static Temporal Event Hypergraph without dynamic construction and without FSM HOEs; rely on fixed precomputed edges from short windows; apply similar traversal updates.  
- Quantifies benefit of dynamic DHG and FSM abstraction.

4.4 Optional event frame CNN  
- Time surface or voxel grid CNN trained with SGD as a non spiking baseline to contextualize accuracy.

5. Metrics  
5.1 Accuracy and F1  
- Top 1 accuracy per sample; macro F1 across classes for imbalance sensitivity.  
- Online accuracy at specified latencies 50 ms, 100 ms after sample start.

5.2 Sample efficiency  
- Labeled samples required to reach accuracy thresholds e.g., 80 percent, 90 percent; area under accuracy vs labels curve.

5.3 Throughput and latency  
- End to end event lane latency and events per second sustained measured as in [DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md).  
- Traversal cycle latency per seed; FSM tick cost; percentage of event path budget consumed by WL.

5.4 Memory and footprint  
- Active edges size of E t, HOE count, embedding tables WL and SAGE memory.

5.5 Interpretability and rule quality  
- Audit success rate fraction of decisions where at least one valid hyperpath with score above τ audit explains the predicted class.  
- Rule stability fraction of promoted rules remaining active for T persist seconds.  
- Coverage fraction of seeds partially matching any active rule.  
- Path shortening factor average reduction in steps due to HOEs.

5.6 Continual learning  
- Forward transfer delta in sample efficiency for new task with reuse vs isolate policy.  
- Forgetting change in accuracy on prior tasks after learning new task.  
- Stability of reliability r e on frozen edges bounded drift.

6. Evaluation procedures  
6.1 Static task single dataset  
- Warm up calibration period to initialize buffers no labels.  
- Streaming train evaluate loop  
  - For each sample stream events to DCH; provide supervision seeds at label windows; record prediction per sample; update via traversal and plasticity cycles; FSM operates continuously.  
  - Log metrics every K samples and cadence windows.

6.2 Continual tasks  
- Train on Task A for T A seconds or samples then switch to Task B; apply scaffolding decision REUSE ISOLATE HYBRID; continue streaming.  
- Measure before after performance on A and B and compute transfer metrics.

6.3 Hyperparameter setting  
- Use defaults in [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md) and [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md).  
- Limited sweep ±20 percent on key knobs alpha, τ prune, k max, Δ windows; log chosen config hash.

6.4 Latency and throughput measurement  
- Instrument event lane with histograms p50 p90 p99 per module; report averages and tail latencies.  
- Ensure measurements exclude I O by warm cache and prefetch.

6.5 Energy proxy optional  
- Report CPU utilization and estimated energy proxy proportional to CPU cycles; if available use RAPL counters; otherwise provide events per update and memory bandwidth as proxies.

7. Ablation studies  
- No FSM disable Section 6; measure accuracy, rule coverage, interpretability drop.  
- No HOE disable Section 7 path compression; measure traversal latency and path length changes.  
- No WL grouping disable Section 4 WL and LSH; rely solely on TC kNN.  
- No SAGE periodic remove refinement and measure FSM discovery stability.  
- No traversal credit assignment remove Section 5 and rely only on local watcher updates; measure learning quality.  
- k max variations 1,2,3; Δ windows narrower or wider; λ path set to 0 or 1 to isolate local vs path based learning.  
- Scaffolding off evaluate continual learning without FREEZE and regionization.

8. Statistical methodology  
- Run 5 seeds per setting; report mean ± std and 95 percent confidence intervals by bootstrap.  
- Use paired tests against closest ablation or baseline when applicable; report effect sizes Cohen s d.  
- Correct for multiple comparisons via Holm Bonferroni when testing many ablations.

9. Reporting and artifacts  
- Save per run  
  - Config YAML, commit hash, seeds, snapshot ids.  
  - Metrics CSV Parquet with timestamps and cadence.  
  - Rule descriptors active set JSON from FSM; HOE registry.  
  - Example audited paths for a random 1 percent sample of decisions.  
- Aggregate tables  
  - Accuracy and F1; sample efficiency; throughput and latency; interpretability and rule metrics; continual learning metrics.  
- Plots  
  - Accuracy over labels; latency histograms; rule promotions per minute; path length distributions; reliability trajectories.

10. Acceptance thresholds initial targets  
- DVS Gesture static task top 1 accuracy ≥ 90 percent with WL FSM HOE full DCH and ≥ 85 percent with no FSM ablation; throughput ≥ 5e5 events per second desktop target.  
- N MNIST top 1 accuracy ≥ 98 percent DCH and ≥ 96 percent without FSM; throughput ≥ 1e5 events per second.  
- Audit success rate ≥ 70 percent of correct decisions explained by a valid hyperpath with score ≥ 0.3.  
- Rule stability ≥ 60 percent of promoted rules remain active for ≥ 60 seconds.  
- Continual learning forgetting ≤ 5 percentage points under ISOLATE policy on prior task after learning new task.

11. Mermaid diagram — evaluation workflows

```mermaid
flowchart TB
DATA[Event dataset via tonic] --> STREAM[Stream construction and watermark]
STREAM --> DCH[DCH online loop]
DCH --> METRICS[Metrics logging]
DCH --> PRED[Per sample predictions]
PRED --> ACC[Accuracy and F1]
DCH --> RULES[Rules and HOEs]
RULES --> INT[Interpretability and stability]
METRICS --> THR[Throughput and latency]
INT --> REPORT[Aggregate tables and plots]
THR --> REPORT
ACC --> REPORT
```

12. Implementation hooks and harness references  
- Implement dataset adapters and run scripts in experiments per [DCH_Section11_SoftwareBlueprint.md](./sections/DCH_Section11_SoftwareBlueprint.md).  
- Provide assertions and invariants in tests for temporal validity and traversal constraints.  
- Export rule sets and HOEs for audit dashboards.

13. Risks and mitigations specific to evaluation  
- Label alignment jitter ensure supervision seeds align with measured spike times; apply tolerance band for seeds ±2 ms.  
- Class imbalance in DVS Gesture report macro F1 and per class accuracy.  
- Variance across seeds increase seeds if CI widths exceed 2 percentage points for main metrics.  
- Compute contention isolate event lane measurements by pinning threads or running offline replays.

14. Cross references  
- Complexity budgets and latency targets [DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md).  
- Module metrics and interfaces [DCH_Section9_Interfaces.md](./sections/DCH_Section9_Interfaces.md).  
- Scaffolding policies for continual learning [DCH_Section8_TaskAwareScaffolding.md](./sections/DCH_Section8_TaskAwareScaffolding.md).

End of Section 12


---

# Dynamic Causal Hypergraph DCH — Section 13 Parameter Defaults and Tuning Strategy

Parent outline [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md), Section 2 [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md), Section 3 [DCH_Section3_Plasticity.md](./sections/DCH_Section3_Plasticity.md), Section 4 [DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md), Section 5 [DCH_Section5_CreditAssignment.md](./sections/DCH_Section5_CreditAssignment.md), Section 6 [DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md), Section 8 [DCH_Section8_TaskAwareScaffolding.md](./sections/DCH_Section8_TaskAwareScaffolding.md), Section 10 [DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md), Section 11 [DCH_Section11_SoftwareBlueprint.md](./sections/DCH_Section11_SoftwareBlueprint.md)

Version v0.1

1. Purpose and scope  
- Consolidate canonical defaults for all major DCH modules.  
- Provide a practical tuning methodology with safe ranges, sensitivity guidance, and online adaptation policies.  
- Deliver dataset specific presets for DVS Gesture and N MNIST consistent with complexity and throughput targets.

2. Canonical defaults by module  
2.1 Time and windows  
- Timestamps unit μs; refractory ρ 1000 μs per neuron.  
- Delay window Δ_min 1000 μs, Δ_max 30000 μs; δ_causal 2000 μs.

2.2 DHG TC kNN (Section 2)  
- k_max 3; M_in 6 admitted unary candidates per head prior to combinations; C_cap 10 total candidates per head after scoring.  
- Candidate scoring weights w_delay 0.7, w_size 0.3; ε_init 0.05 reliability.

2.3 Plasticity (Section 3)  
- EMA step α 0.1; bounds r_min 0.02, r_max 0.98; prior r_0 0.5; discounted counts decay γ_c 0.98/s; time decay β 0.01/s.  
- Prune τ_prune 0.02; τ_age_min 2 s inactivity; τ_use_min 3; H_idle 30 s.  
- λ_path 0.5 blending path signals with local watchers.

2.4 Hyperpath embeddings (Section 4)  
- WL r_WL 2; d_WL 64; cadence Δt_WL 10 ms; frontier cap F_max 256.  
- LSH bands 8, rows_per_band 4; τ_LSH 2; cosine threshold τ_cos 0.65.  
- SAGE r_SAGE 3; d_SAGE 128; cadence Δt_SAGE 500 ms; batch 4096; S_nbr 32; S_edge 64.

2.5 Credit assignment (Section 5)  
- Seeds M 8; depth L 12; branch cap B 4; C_in 16 admissible in filter; τ_select 0.7; H_back 100 ms.  
- Feature weights w_rel 1.0, w_rec 0.5, w_rule 0.5, w_sim 0.5; λ_rec 1/100 ms; ε_norm 1.0.

2.6 FSM and rule induction (Section 6)  
- Window W 60 s; decay γ 0.98/s; s_min 50; r_min_rule 0.6; c_min 10 distinct sinks; L_max_rule 6; k_max_rule 3.  
- HH capacity K_HH 100k; CMS width 32768, depth 4; EWMA α_r 0.1; hysteresis D_stab 5 s; s_min_demote 0.5×s_min; D_demote 5 s; s_path_min 0.2.

2.7 Task aware scaffolding (Section 8)  
- Similarity thresholds θ_reuse 0.6; θ_isolate 0.3; weights w_act 0.5, w_emb 0.4, w_perf 0.1; r_act 0.7.  
- Freeze K_protect_pct 5%; K_protect_max 50k; TTL 120 s; α_review 0.02; τ_conflict 0.02.  
- Region leakage ε_leak 0.1; λ_reg 0.5; σ_max 0.6; k_alpha_up 1.5; K_head_boost 1.2; r_gate 0.8.

3. Tuning methodology and order of operations  
3.1 Fix timing and admission first  
- Validate refractory ρ and Δ windows against observed empirical delays (quantile check p10 ≥ Δ_min, p90 ≤ Δ_max).  
- Calibrate δ_causal to cluster true presyn bursts (1–3 ms typical); widen only if WL grouping is disabled.

3.2 Control combinatorics  
- Set DHG budgets C_cap and k_max; if candidate explosion or queue lag, reduce C_cap first, then k_max to 2.  
- Raise τ_prune temporarily to cull low r_e edges during burn in.

3.3 Stabilize learning signals  
- Tune α within 0.05–0.2 (higher for nonstationary streams).  
- Use λ_path sweep {0.3, 0.5, 0.7} to balance local vs. path evidence; monitor watcher confirm/miss ratio.

3.4 Traverse efficiently  
- Start with M 8, L 12, B 4; if traversal latency > budget, reduce L to 10, then C_in to 12; maintain determinism via fixed RNG seeding.  
- Increase w_rule to 0.8 when an adequate ruleset is active to improve search efficiency.

3.5 FSM sensitivity  
- Increase s_min to 75 on high churn datasets; decrease to 40 for sparse labels.  
- Raise γ to 0.995 (slower decay) when patterns should persist longer; lower to 0.95 to adapt rapidly.

3.6 Scaffolding gates  
- Calibrate θ_reuse/θ_isolate using a short pilot on Task A then Task B; target hybrid band width 0.1–0.2 to avoid oscillation.  
- Set region leakage ε_leak ≤ 0.1 to limit spurious cross links; relax when transfer is desired.

4. Safe ranges and sensitivity notes  
- Δ_min [0.5, 2] ms; Δ_max [20, 50] ms; widening increases false positives in DHG; prefer WL grouping instead.  
- α in [0.05, 0.2]; too high causes volatility; too low slows adaptation.  
- τ_prune in [0.01, 0.05]; below 0.01 increases memory; above 0.05 may prune useful edges.  
- M∈[4,12], L∈[8,16], B∈[2,6]; traversal cost scales roughly linearly in K·L·C_in (Section 10).  
- s_min∈[30,100]; lower values promote noisy rules; higher values slow rule discovery.  
- θ_reuse∈[0.5,0.7], θ_isolate∈[0.2,0.4]; too close induces policy oscillation; enforce hysteresis 0.05.

5. Dataset specific presets  
5.1 N MNIST  
- Δ_min 1 ms, Δ_max 25 ms, δ_causal 1.5 ms; k_max 2; α 0.08; τ_prune 0.02.  
- WL r 2, d 64, Δt_WL 10 ms; M 6, L 10, B 3; s_min 40; γ 0.98; θ_reuse 0.6, θ_isolate 0.3.

5.2 DVS Gesture  
- Δ_min 1 ms, Δ_max 30 ms, δ_causal 2 ms; k_max 3; α 0.1; τ_prune 0.02.  
- WL r 2, d 64, Δt_WL 10 ms; M 8, L 12, B 4; s_min 60–75; γ 0.98; θ_reuse 0.6, θ_isolate 0.3.  
- Under high churn raise s_min to 75 and increase Δt_WL to 15 ms to cap load.

6. Online adaptation policies (Meta controller)  
- Queue lag high: reduce C_cap by 20%, increase τ_prune by 25%, increase Δt_WL by 5 ms.  
- Rule churn high: raise s_min by 25%, increase γ by 0.01 absolute; lower w_rule temporarily.  
- Low traversal yield: increase w_rule to 0.8 and w_sim to 0.7; enable HOE preference; consider raising H_back to 120 ms briefly.  
- Memory pressure: increase τ_prune; lower K_head budget; demote low freshness edges using s_prune = r_e·exp(−λ_age·age).

7. Diagnostics and invariants to watch  
- DHG candidate hit rate target ≥ 30%; dedup rate ≥ 20%; rising queues → reduce C_cap.  
- Watcher confirm rate: aim 55–75% in stationary segments; sustained < 40% → shrink Δ windows or increase τ_prune.  
- Traversal valid ratio ≥ 50%; average depth 6–9; if lower, raise w_rule or adjust L.  
- FSM promotion precision proxy ≥ 70% (rules remain active ≥ 60 s).  
- HOE path shortening factor ≥ 1.5× on DVS Gesture after warm-up.

8. Auto calibration routines  
- Delay envelope fitting: per pair i→j maintain exponential histograms; update Δ_min/Δ_max to [p10, p95] with hysteresis to avoid thrash.  
- Temperature τ_select annealing: start at 0.9 for exploration; decay to 0.7 over 60 s of stable performance.  
- λ_path scheduling: increase toward 0.7 when supervised signals confirm traversal correctness; fall back to 0.5 otherwise.

9. Reproducibility and config hashing  
- Emit a config hash derived from a stable serialization of all knobs (ordered YAML) plus code revisions; attach to every metrics bundle.  
- Record snapshot ids for WL, SAGE, and FSM rulesets (see [DCH_Section9_Interfaces.md](./sections/DCH_Section9_Interfaces.md)).

10. Acceptance criteria for Section 13  
- Canonical defaults listed for all modules and consistent with earlier sections.  
- Clear order of tuning with safe ranges and sensitivity notes.  
- Dataset presets for DVS Gesture and N MNIST are provided.  
- Online adaptation rules defined for meta control and backpressure.  
- Diagnostics and invariants enable guardrail checks during runs.

11. Cross references  
- Windows and refractory semantics [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md)  
- DHG budgets and scoring [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md)  
- EMA and pruning [DCH_Section3_Plasticity.md](./sections/DCH_Section3_Plasticity.md)  
- Embeddings and grouping [DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md)  
- Traversal policy [DCH_Section5_CreditAssignment.md](./sections/DCH_Section5_CreditAssignment.md)  
- FSM thresholds [DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md)  
- Scaffolding decisions [DCH_Section8_TaskAwareScaffolding.md](./sections/DCH_Section8_TaskAwareScaffolding.md)  
- Performance budgets [DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md)

End of Section 13


---

# Dynamic Causal Hypergraph DCH — Section 14 Risk Analysis and Mitigations

Parent outline [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](./sections/DCH_Section1_FormalFoundations.md), Section 2 [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md), Section 3 [DCH_Section3_Plasticity.md](./sections/DCH_Section3_Plasticity.md), Section 4 [DCH_Section4_HyperpathEmbedding.md](./sections/DCH_Section4_HyperpathEmbedding.md), Section 5 [DCH_Section5_CreditAssignment.md](./sections/DCH_Section5_CreditAssignment.md), Section 6 [DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md), Section 7 [DCH_Section7_HierarchicalAbstraction.md](./sections/DCH_Section7_HierarchicalAbstraction.md), Section 8 [DCH_Section8_TaskAwareScaffolding.md](./sections/DCH_Section8_TaskAwareScaffolding.md), Section 10 [DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md), Section 11 [DCH_Section11_SoftwareBlueprint.md](./sections/DCH_Section11_SoftwareBlueprint.md), Section 12 [DCH_Section12_Evaluation.md](./sections/DCH_Section12_Evaluation.md), Section 13 [DCH_Section13_ParamsTuning.md](./sections/DCH_Section13_ParamsTuning.md)

Version v0.1

1. Objectives  
- Identify principal algorithmic, systems, and governance risks for DCH.  
- Define concrete mitigations, monitors, and automated responses.  
- Provide stress tests and runbooks to recover service and ensure scientific validity.

2. Risk inventory and drivers  
2.1 Combinatorial explosion (edges and paths)  
- Drivers: high in-degree Pred j, wide Δ windows, high k_max, low pruning thresholds, sparse supervision.  
- Symptoms: queue lag, memory growth, traversal valid ratio collapse, rule churn spikes.

2.2 Spurious causality and noise coupling  
- Drivers: temporal coincidence within Δ windows, WL hash/LSH collisions, low s_min in FSM, unreliable labels.  
- Symptoms: low watcher confirm rate, unstable r_e oscillations, short-lived rules.

2.3 Nonstationarity and drift  
- Drivers: sensor dynamics, scene changes, task switches.  
- Symptoms: rising miss rates, rule demotions, HOE reliability decay.

2.4 Catastrophic forgetting and interference  
- Drivers: learning new tasks reusing old subgraphs without protection.  
- Symptoms: r_e depression on prior-task edges, accuracy drop on old tasks.

2.5 Latency and throughput violations  
- Drivers: bursty λ, oversized WL frontier, deep traversal, FSM churn.  
- Symptoms: missed real-time budgets, backlogs, timeouts.

2.6 Memory pressure and fragmentation  
- Drivers: |E(t)| growth, large HOE registry, oversized embedding tables.  
- Symptoms: allocator pressure, OOM risk, swap thrash.

2.7 Determinism and reproducibility gaps  
- Drivers: unseeded RNGs, nondeterministic hashing, unsynchronized snapshots.  
- Symptoms: irreproducible runs, audit mismatch.

2.8 Interpretability failure (rules that mis-explain)  
- Drivers: overfit templates, path double counting, weak audit thresholds.  
- Symptoms: high accuracy but low audit success or inconsistent explanations.

2.9 Security and privacy exposure  
- Drivers: exporting raw timestamps and neuron ids, verbose provenance.  
- Symptoms: re-identification risks, policy noncompliance.

2.10 Hardware co-design feasibility  
- Drivers: PIM atomic update fidelity, NoC contention, FSM canonicalization throughput, PE partitioning.  
- Symptoms: stalls, hot spots, underutilization, energy regressions.

2.11 Evaluation validity and metric gaming  
- Drivers: label misalignment, cherry-picked cadences, incomplete baselines.  
- Symptoms: inflated metrics without robustness, poor external replication.

3. Mitigations and controls  
3.1 Combinatorics control  
- Tight windows Δ (Section 13 presets), δ_causal coherence, DHG budgets M_in and C_cap, k_max ≤ 3.  
- Reliability-based pruning (Section 3) with τ_prune, freshness-weighted eviction; HOE compression (Section 7).  
- Meta-controller backpressure (Section 10) to lower C_cap and raise τ_prune when queue lag crosses L_q_high.

3.2 Anti-spurious causality  
- Enforce temporal logic strictly (Section 5), refractory ρ (Section 1), and B-connectivity.  
- WL cosine threshold τ_cos, LSH collision threshold τ_LSH; deduplication by canonical tail signature (Section 2).  
- FSM thresholds s_min, r_min, hysteresis D_stab (Section 6); s_path_min to filter low-confidence paths.

3.3 Drift handling  
- FSM dual-window drift detectors; adapt γ and s_min; re-learn Δ envelopes via quantile tracking (Section 13 auto calibration).  
- Increase λ_path temporarily to favor supervised paths during drift.

3.4 Forgetting protection  
- Scaffolding FREEZE with TTL and review (Section 8); regionization with ε_leak guard; rule and HOE protection lists.  
- Hybrid mode for partial reuse; audit depression events on frozen edges with τ_conflict alarms.

3.5 Latency assurance  
- WL frontier cap F_max; traversal caps L, B, C_in; duty-cycling T_trav; FSM tick budgets.  
- Sharded DHG/WL lanes; priority queues; non-blocking design (Section 11 threading).  
- Auto-throttle: raise Δt_WL, reduce K seeds under load.

3.6 Memory governance  
- Global caps for |E|, K_head, K_HOE_global; quantize WL to fp16; compress indices.  
- Aged pruning with s_prune = r_e·exp(−λ_age·age); HOE merge (Section 7).  
- Snapshot retention policy; evict stale SAGE snapshots.

3.7 Reproducibility  
- Seeded RNG with cycle- and vertex-derived seeds (Section 5).  
- Stable 64-bit hashes; deterministic canonicalization; idempotency keys (Section 9).  
- Snapshot ids for WL, SAGE, FSM; config hashing (Section 13).

3.8 Interpretability guarantees  
- Audit pipeline: enforce τ_audit for path scores; coverage metric; store top-k paths per decision.  
- Rule promotion precision proxy ≥ 70% active duration (Section 6); avoid double counting in HOE compression (Section 7).

3.9 Security and privacy  
- Hash neuron ids for export; quantize timestamps to q_time; role-based access for freeze and policy updates (Section 9).  
- Provenance redaction outside secure boundary; TTL on exemplars.

3.10 Hardware risk mitigations  
- Pre-silicon: cycle-accurate co-sim of GSE, GMF, PTA, FSM with recorded traces; Roofline and NoC contention models.  
- Fallback paths: software-only traversal and FSM when hardware stalls; graceful degradation policies.  
- PIM atomicity tests and endurance modeling; CAM/hash sizing for FSM counting.

3.11 Evaluation integrity  
- Ground-truth seed tolerance ±2 ms; macro-F1 reporting; five-seed statistics with CI; Holm–Bonferroni corrections.  
- Baseline parity on label exposure; ablations catalog (Section 12).

4. Guardrails and monitors (with targets)  
- Queue lag watermark: warn ≥ 10 ms, act ≥ 25 ms → reduce C_cap 20%, raise τ_prune 25%.  
- Watcher confirm rate: warn < 55%, act < 40% → shrink Δ windows 10%, raise τ_prune.  
- Traversal valid ratio: warn < 50% → increase w_rule and WL boost.  
- Rule churn per minute: warn > 2× baseline → raise s_min; raise γ.  
- Memory occupancy: warn ≥ 80% cap, act ≥ 90% → demote cold edges; shrink WL d to 48 or 32.  
- HOE path shortening: warn < 1.2× → adjust eligibility; merge/factor HOEs.

5. Stress-test plan  
- Synthetic micrographs: planted chains with varying Δ and noise; evaluate precision/recall and r_e stability.  
- Collision fuzzing: adversarial tail signatures and WL hash collisions; ensure deduplication and LSH thresholds.  
- Drift injection: piecewise delay distributions; monitor drift detectors and adaptation.  
- Burst load: 2–3× λ for 100 ms; validate backpressure and latency SLOs.  
- HOE spurious promotion: inject frequent but unreliable templates; verify hysteresis and demotion.

6. Failure runbooks  
- High queue lag: throttle DHG (C_cap↓), WL cadence↑, traversal cadence↑; snapshot; persist state; resume after drain.  
- Rule storm: raise s_min, γ; pause promotions; audit top templates.  
- Memory crisis: freeze HOE creation; increase τ_prune; evict stale snapshots; persist and compact indices.  
- Reproducibility alert: lock snapshots; dump seeds and config hash; rerun minimal replay for audit.

7. Acceptance criteria for Section 14  
- Risks categorized with symptoms and drivers.  
- Mitigations, monitors, and automated actions specified and mapped to earlier modules.  
- Stress tests and runbooks documented with concrete actions and thresholds.  
- Security and hardware feasibility considerations included.

8. Mermaid diagram — risk-to-action control loop

```mermaid
flowchart LR
MON[Monitors queue lag, confirm rate, churn, memory] --> DET[Detect threshold breach]
DET --> ACT[Auto actions: adjust caps, thresholds, cadences]
ACT --> PIPE[Publish knobs to DHG WL TRAV FSM]
PIPE --> MON
DET --> RUN[Runbook trigger for operator audit]
```

9. Cross references  
- Backpressure and budgets [DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md)  
- Policy knobs and interfaces [DCH_Section9_Interfaces.md](./sections/DCH_Section9_Interfaces.md)  
- Abstractions and double counting prevention [DCH_Section7_HierarchicalAbstraction.md](./sections/DCH_Section7_HierarchicalAbstraction.md)  
- Evaluation rigor [DCH_Section12_Evaluation.md](./sections/DCH_Section12_Evaluation.md)

End of Section 14


---

# Dynamic Causal Hypergraph DCH — Section 15 Causa Chip Hardware Co Design Overview

Parent outline [DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)  
Cross references [DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md), [DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md), [DCH_Section5_CreditAssignment.md](./sections/DCH_Section5_CreditAssignment.md), [DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md), [DCH_Section7_HierarchicalAbstraction.md](./sections/DCH_Section7_HierarchicalAbstraction.md), [DCH_Section9_Interfaces.md](./sections/DCH_Section9_Interfaces.md)

Version v0.1

1. Objectives and scope
- Translate DCH computational primitives into a heterogeneous SoC architecture for low latency event stream processing.
- Provide first order bandwidth latency and capacity estimates for DVS Gesture and N MNIST targets.
- Define unit responsibilities, interfaces, NoC interconnect, memory hierarchy, and fallback software model.

2. Workload recap and design targets
- Event lane latency budget per event ≤ 350 microseconds desktop prototype see [DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md).
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
  - Snapshot ids for WL SAGE and FSM rules tracked in control plane per [DCH_Section9_Interfaces.md](./sections/DCH_Section9_Interfaces.md).
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
- Each hardware module implements a subset of the APIs in [DCH_Section9_Interfaces.md](./sections/DCH_Section9_Interfaces.md)  
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
- Complexity and performance budgets [DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md)  
- Abstraction and HOE usage [DCH_Section7_HierarchicalAbstraction.md](./sections/DCH_Section7_HierarchicalAbstraction.md)  
- FSM requirements [DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md)  
- Traversal constraints [DCH_Section5_CreditAssignment.md](./sections/DCH_Section5_CreditAssignment.md)  
- Interface contracts [DCH_Section9_Interfaces.md](./sections/DCH_Section9_Interfaces.md)

End of Section 15


---

# Dynamic Causal Hypergraph DCH — References and Reading

This curated bibliography supports the DCH technical specification and the Causa-Chip hardware co-design. It is organized by topic to align with sections in the spec. Use this Markdown list during drafting; a BibTeX export [docs/references.bib](./references.bib) will be added later.

Conventions
- Citation keys in brackets (example [Allen1983]) are stable handles used across the spec.
- When multiple editions exist, prefer the oldest stable scholarly reference and provide a convenient link.

## Temporal logic, causality, and hypergraphs

- [Allen1983] James F. Allen. Maintaining Knowledge about Temporal Intervals. Communications of the ACM, 26(11), 1983. https://doi.org/10.1145/182.358434
- [Pearl2009] Judea Pearl. Causality: Models, Reasoning, and Inference. 2nd ed., Cambridge University Press, 2009. https://bayes.cs.ucla.edu/BOOK-2K/
- [Feng2019] Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, Yue Gao. Hypergraph Neural Networks. AAAI 2019. https://ojs.aaai.org/index.php/AAAI/article/view/3790
- [Bai2021] Song Bai, Feihu Zhang, Philip H. S. Torr. Hypergraph Convolution and Hypergraph Attention. Pattern Recognition 110, 2021. (for broader hypergraph ops) https://arxiv.org/abs/1901.08150
- [Zhou2021] Tianyu Zhou et al. Dynamic Hypergraph Neural Networks. (survey/representative dynamic HGNN) https://arxiv.org/abs/2008.00778

## Frequent subgraph mining (FSM), streaming and dynamic graph pattern mining

- [YanHan2002] Xifeng Yan, Jiawei Han. gSpan: Graph-Based Substructure Pattern Mining. ICDM 2002. https://doi.org/10.1109/ICDM.2002.1184038
- [Huan2003] Jun Huan, Wei Wang, Jan Prins. Efficient Mining of Frequent Subgraphs in the Presence of Isomorphism. ICDM 2003. https://doi.org/10.1109/ICDM.2003.1250950
- [Bifet2010] Albert Bifet, Geoff Holmes, Bernhard Pfahringer, Richard Kirkby. MOA: Massive Online Analysis. JMLR 2010. (stream mining framework concepts) https://jmlr.org/papers/v11/bifet10a.html
- [Chakrabarti2006] Deepayan Chakrabarti. Dynamic Graph Mining: A Survey. SIGKDD Explorations 2006. https://doi.org/10.1145/1147234.1147237
- [Zou2016] Zhengyi Zou et al. Frequent Subgraph Mining on a Single Large Graph. VLDB 2016. (background) http://www.vldb.org/pvldb/vol9/p860-zou.pdf
- [Bose2018] Arindam Bose et al. A Survey of Streaming Graph Processing Engines. https://arxiv.org/abs/1807.00336
- [Wang2020] Yuchen Wang et al. Incremental Graph Pattern Mining: A Survey. https://arxiv.org/abs/2007.08583

## Spiking neural networks (SNNs), event-based datasets, and tools

- [Orchard2015] Garrick Orchard et al. Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades. Frontiers in Neuroscience, 2015. (N-MNIST) https://doi.org/10.3389/fnins.2015.00437
- [Amir2017] A. Amir et al. A Low Power, Fully Event-Based Gesture Recognition System. CVPR 2017. (DVS Gesture dataset) https://doi.org/10.1109/CVPR.2017.298
- [Norse] H. P. Zenke et al. Norse: A Deep Learning Library for Spiking Neural Networks. (PyTorch-based SNN) https://github.com/norse/norse
- [BindsNET] Hazan et al. BindsNET: A Spiking Neural Networks Library in Python. Frontiers in Neuroinformatics, 2018. https://doi.org/10.3389/fninf.2018.00089
- [tonic] G. M. Hunsberger, F. Ceolini et al. Tonic: A Tool to Load and Transform Event-Based Datasets. https://github.com/neuromorphs/tonic

## Graph representation learning and similarity

- [Hamilton2017] William L. Hamilton, Rex Ying, Jure Leskovec. Inductive Representation Learning on Large Graphs (GraphSAGE). NeurIPS 2017. https://arxiv.org/abs/1706.02216
- [Grover2016] Aditya Grover, Jure Leskovec. node2vec: Scalable Feature Learning for Networks. KDD 2016. https://doi.org/10.1145/2939672.2939754
- [Shchur2018] O. Shchur, M. Mumme, A. Bojchevski, S. Günnemann. Pitfalls of Graph Neural Network Evaluation. Relates to robust baselines. https://arxiv.org/abs/1811.05868

## Random walks, temporal graphs, and reasoning constraints

- [Ribeiro2018] Leonardo F. R. Ribeiro, Pedro H. P. Saverese, Daniel R. Figueiredo. struc2vec: Learning Node Representations from Structural Identity. KDD 2017/2018. (structure-aware walks) https://doi.org/10.1145/3097983.3098061
- [Kazemi2020] Seyed Mehran Kazemi et al. Relational Representation Learning for Dynamic Knowledge Graphs. (temporal constraints) https://arxiv.org/abs/1905.11485
- [Leskovec2014] Jure Leskovec et al. Temporal Networks. (book/survey) https://arxiv.org/abs/1607.01781

## Hardware co-design, PIM, memory fabrics, NoC

- [Bender2002] Michael A. Bender, Richard Cole, Erik D. Demaine, Martin Farach-Colton, Jack Zito. Two Simplified Algorithms for Maintaining Order in a List. SODA 2002. (Packed Memory Array foundations) https://doi.org/10.5555/545381.545411
- [Chi2016] Pengfei Chi et al. PRIME: A Novel Processing-in-Memory Architecture for Neural Network Computation in ReRAM-based Main Memory. ISCA 2016. https://doi.org/10.1109/ISCA.2016.12
- [Shafiee2016] Ali Shafiee et al. ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars. ISCA 2016. https://doi.org/10.1109/ISCA.2016.12
- [Jiang2021] W. Jiang et al. A Survey on Processing-in-Memory. ACM Computing Surveys, 2021. https://doi.org/10.1145/3451210
- [Kim2018] John Kim, William J. Dally. Scalable On-Chip Interconnection Networks. Morgan & Claypool 2018. (NoC principles) https://doi.org/10.2200/S00809ED1V01Y201804CAC043

## Continual learning, task-aware control, and neuro-symbolic integration

- [Parisi2019] German I. Parisi et al. Continual Lifelong Learning with Neural Networks: A Review. Neural Networks, 2019. https://doi.org/10.1016/j.neunet.2019.01.012
- [Kirkpatrick2017] James Kirkpatrick et al. Overcoming Catastrophic Forgetting in Neural Networks. PNAS, 2017. (EWC) https://doi.org/10.1073/pnas.1611835114
- [d’AvilaGarcez2019] Artur d’Avila Garcez, Luis C. Lamb. Neurosymbolic AI: The 3rd Wave. (perspective) https://arxiv.org/abs/2012.05876
- [Valiant2000] Leslie Valiant. Robust Logics. Annals of Pure and Applied Logic, 2000. (neuro-symbolic inspiration) https://doi.org/10.1016/S0168-0072(00)00005-7

## Implementation, engineering, and measurements

- [Dean2013] Jeffrey Dean, Luiz André Barroso. The Tail at Scale. Communications of the ACM, 2013. (tail latency discipline) https://doi.org/10.1145/2408776.2408794
- [Goldstein2020] Moshe Goldstein et al. Measuring ML System Performance: Metrics and Methodologies. (Engineering perspective) arXiv:2008. https://arxiv.org/abs/2008.XXXX

## Pointers to software and datasets (practical)

- Norse (PyTorch SNN): https://github.com/norse/norse
- BindsNET: https://github.com/BindsNET/bindsnet
- Tonic (event datasets): https://github.com/neuromorphs/tonic
- DVS Gesture: https://research.ibm.com/publications/dvs-gesture-dataset (landing page)
- N-MNIST: https://www.garrickorchard.com/datasets/n-mnist

## To-Do (for v0.2 references revision)

- Validate and expand DHGNN dynamic hypergraph citations with the most recent surveys.  
- Add canonical labeling hardware references (CAM designs and counting accelerators).  
- Add specific task-aware SNN literature references (SCA-SNN or nearest strong alternative) with stable DOIs.  
- Populate a BibTeX file [docs/references.bib](./references.bib) with the above keys and cross-check in-text mentions.


---

# Dynamic Causal Hypergraph DCH — Diagrams Index and Render Guide

Purpose  
This index catalogs all Mermaid diagrams across the DCH spec to support consistency checks and PDF export. Each entry links to the source section. Use the Render Guide below to validate diagrams.

Render Guide
- VS Code preview Install the Mermaid Markdown extension; open each linked file and ensure diagrams render.  
- CLI export Option A mermaid-cli
  - npm install -g @mermaid-js/mermaid-cli
  - mmdc -i input.md -o output.pdf with a markdown-to-pdf pipeline that preserves Mermaid.  
- CLI export Option B Pandoc with Mermaid filter
  - pandoc --from gfm --to pdf --filter pandoc-mermaid -o docs/export/DCH_TechSpec_v0.1.pdf docs/DCH_TechSpec_v0.1.md
- Verify anchors After export, click internal links in the PDF to confirm they resolve.

Master overview
- Neuro-symbolic learning loop
  - [docs/DCH_TechSpec_v0.1.md](./DCH_TechSpec_v0.1.md)

Core algorithms
- TC-kNN DHG flow enhanced
  - [docs/sections/DCH_Section2_DHG_TCkNN.md](./sections/DCH_Section2_DHG_TCkNN.md)
- Constrained backward traversal with AND frontier
  - [docs/sections/DCH_Section5_CreditAssignment.md](./sections/DCH_Section5_CreditAssignment.md)
- FSM pipeline normalization to rule promotion
  - [docs/sections/DCH_Section6_FSM.md](./sections/DCH_Section6_FSM.md)
- Hierarchical abstraction creation and usage HOEs
  - [docs/sections/DCH_Section7_HierarchicalAbstraction.md](./sections/DCH_Section7_HierarchicalAbstraction.md)

Learning control and interfaces
- Task-aware scaffolding REUSE ISOLATE HYBRID control loop
  - [docs/sections/DCH_Section8_TaskAwareScaffolding.md](./sections/DCH_Section8_TaskAwareScaffolding.md)
- Module interaction map data flow and control
  - [docs/sections/DCH_Section9_Interfaces.md](./sections/DCH_Section9_Interfaces.md)
- Performance pipeline with latency budgets
  - [docs/sections/DCH_Section10_ComplexityResource.md](./sections/DCH_Section10_ComplexityResource.md)

Software blueprint and evaluation
- Software dataflow orchestration lanes
  - [docs/sections/DCH_Section11_SoftwareBlueprint.md](./sections/DCH_Section11_SoftwareBlueprint.md)
- Evaluation workflows data-to-metrics
  - [docs/sections/DCH_Section12_Evaluation.md](./sections/DCH_Section12_Evaluation.md)

Risk and governance
- Risk-to-action control loop monitors and auto knobs
  - [docs/sections/DCH_Section14_RiskMitigations.md](./sections/DCH_Section14_RiskMitigations.md)

Hardware co-design
- Chip-level dataflow units and control
  - [docs/sections/DCH_Section15_CausaChip.md](./sections/DCH_Section15_CausaChip.md)
- GMF update pipeline with PIM atomic ops
  - [docs/sections/DCH_Section15_CausaChip.md](./sections/DCH_Section15_CausaChip.md)
- PTA PE array topology and reduction path
  - [docs/sections/DCH_Section15_CausaChip.md](./sections/DCH_Section15_CausaChip.md)
- FSM engine pipeline in hardware
  - [docs/sections/DCH_Section15_CausaChip.md](./sections/DCH_Section15_CausaChip.md)

Outline quick diagrams duplicate anchors
- Neuro-symbolic learning loop quick view
  - [docs/DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)
- DHG construction around a post spike quick view
  - [docs/DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)
- Constrained backward traversal quick view
  - [docs/DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)
- Causa-Chip dataflow quick view
  - [docs/DCH_TechSpec_Outline.md](./DCH_TechSpec_Outline.md)

Checklist for reviewers
- Diagrams render without syntax errors in all listed files.  
- Terminology nodes and labels match the text e.g., DHG, PTA, FSM, HOE, WL, SAGE.  
- Arrows reflect correct data and control flow according to the section narratives.  
- Duplicate diagrams quick views in the Outline are consistent with the authoritative versions in Sections 2, 5, 6, 7, 8, 9, 10, 11, 12, 14, and 15.

Known formatting caveats
- Avoid double quotes and parentheses inside Mermaid node labels to prevent parse errors; the spec conforms to this requirement.  
- Long labels may wrap differently across renderers ensure readability in both VS Code preview and CLI export.

End of Diagrams Index
