# Dynamic Causal Hypergraph DCH — Section 7 Hierarchical Abstraction and Higher Order Hyperedges

Parent outline [DCH_TechSpec_Outline.md](../DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](../sections/DCH_Section1_FormalFoundations.md), Section 2 [DCH_Section2_DHG_TCkNN.md](../sections/DCH_Section2_DHG_TCkNN.md), Section 3 [DCH_Section3_Plasticity.md](../sections/DCH_Section3_Plasticity.md), Section 6 [DCH_Section6_FSM.md](../sections/DCH_Section6_FSM.md)

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
- Rule promotion and descriptors Section 6 [DCH_Section6_FSM.md](../sections/DCH_Section6_FSM.md)  
- DHG candidate generation Section 2 [DCH_Section2_DHG_TCkNN.md](../sections/DCH_Section2_DHG_TCkNN.md)  
- EMA reliability and pruning Section 3 [DCH_Section3_Plasticity.md](../sections/DCH_Section3_Plasticity.md)

End of Section 7