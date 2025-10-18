# Dynamic Causal Hypergraph DCH — Section 6 Streaming Frequent Hyperpath Mining and Online Rule Induction

Parent outline [DCH_TechSpec_Outline.md](../DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](../sections/DCH_Section1_FormalFoundations.md), Section 4 [DCH_Section4_HyperpathEmbedding.md](../sections/DCH_Section4_HyperpathEmbedding.md), Section 5 [DCH_Section5_CreditAssignment.md](../sections/DCH_Section5_CreditAssignment.md)

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
- Formal path and reliability definitions Section 1 [DCH_Section1_FormalFoundations.md](../sections/DCH_Section1_FormalFoundations.md)  
- Embedding integration Section 4 [DCH_Section4_HyperpathEmbedding.md](../sections/DCH_Section4_HyperpathEmbedding.md)  
- Traversal source and priors Section 5 [DCH_Section5_CreditAssignment.md](../sections/DCH_Section5_CreditAssignment.md)  
- Abstraction sink Section 7 (to be drafted) [DCH_TechSpec_Outline.md](../DCH_TechSpec_Outline.md)

End of Section 6