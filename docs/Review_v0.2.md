# DCH v0.2 — Internal Review Package

Purpose
- Drive the internal review for v0.2 of the Dynamic Causal Hypergraph (DCH) specification and code plan.
- Provide reviewer checklist, automated validation results, findings, proposed edits, and a sign‑off rubric.

Scope of review (documents and code artifacts)
- Master spec: docs/DCH_TechSpec_v0.1.md
- Assembled one‑file spec: docs/DCH_TechSpec_v0.1_assembled.md
- Printable HTML artifact (render + print to PDF): docs/export/DCH_TechSpec_v0.1.html
- Diagrams index and report: docs/DiagramsIndex.md, docs/DiagramsReport.md
- Links validation report: docs/LinksReport.md
- Go/No-Go brief: docs/GoNoGo.md
- Prototype modules (heads‑up only for context): dch_core/* and tests/*

Automated validation summary (current state)
- Diagrams (Mermaid): PASS (0 issues)
  - Validator: scripts/check_diagrams.py
  - Report: docs/DiagramsReport.md
- Links and anchors across repo: PASS (0 issues)
  - Validator: scripts/check_links.py
  - Latest report: docs/LinksReport.md
- Integration tests: PASS
  - tests/test_integration_smoke.py — OK
  - tests/test_fsm_minimal.py — OK
  - tests/synthetic_micro.py — OK

Reviewer checklist (tick each when verified)
1. Formalism and invariants
   - [ ] Section 1 definitions V(t), E(t), hyperedge attributes (Δmin, Δmax, ρ, r) are precise and consistent with usage.
   - [ ] B‑connectivity semantics explicitly constrain backward traversal; examples reflect AND frontier semantics.
   - [ ] Invariants and safety (reliability bounds, no head clashes < ρ, acyclicity under ABSTRACT) are stated and cross‑referenced.

2. Algorithms — DHG construction (TC‑kNN)
   - [ ] Presyn source discovery (Pred(j)) and temporal windows align with defaults (1–30 ms; δ_causal = 2 ms).
   - [ ] Candidate generation includes unary and multi‑tail within δ_causal; dedup and budgets specified.
   - [ ] Initialization and admission policies are justified; complexity and data structures described.

3. Algorithms — Plasticity and pruning
   - [ ] Predict/confirm/miss watcher model specified; discounted counts and EMA rule (α=0.1, clamp [0.02,0.98]) match Section 3.
   - [ ] Pruning thresholds and “use it or lose it” policy control growth; housekeeping documented.
   - [ ] Concurrency determinism called out (update order, timestamps).

4. Algorithms — Credit assignment traversal
   - [ ] Multi‑start randomized beam traversal parameters (M,L,B,τ_select,H_back) present and reasonable.
   - [ ] Temporal logic constraints (Δ windows, refractory) enforced in traversal; B‑connectivity implemented.
   - [ ] Path score aggregation and contribution semantics unambiguous.

5. Algorithms — Embeddings and grouping
   - [ ] WL streaming embedding role (online grouping) vs. periodic SAGE refinement separated and justified.
   - [ ] Causal‑context similarity used only to group/guide, not to assert causation.

6. Algorithms — Streaming FSM and rules
   - [ ] Canonicalization, windowed heavy hitters/CMS, thresholds (W, s_min, r_min, γ) consistent and motivated.
   - [ ] Promotion produces RuleDescriptor and integrates with HOE abstraction.

7. Hierarchical abstraction (HOEs) and scaffolding
   - [ ] HOE creation constraints prevent cycles/duplication; provenance stored.
   - [ ] Task‑aware scaffolding (REUSE/ISOLATE/HYBRID) policy parameters and freeze semantics defined; regionization clear.

8. Interfaces and complexity/resource model
   - [ ] Section 9 interfaces map cleanly to prototype structures.
   - [ ] Section 10 latency/memory targets achievable under stated workloads (DVS Gesture, N‑MNIST).

9. Evaluation and tuning
   - [ ] Datasets, metrics, ablations, acceptance thresholds enumerated and feasible.
   - [ ] Tuning defaults and safe ranges cross‑referenced from Section 13.

10. Risks and hardware co‑design
   - [ ] Risk/runbooks present (combinatorics, drift, forgetting, latency).
   - [ ] Causa‑Chip mapping coherent: GSE, GMF, PTA, FSM, MC; dataflows align with SLOs.

11. Export readiness
   - [ ] Printable HTML artifact renders all Mermaid diagrams and internal links.
   - [ ] PDF export pathway documented (pandoc or browser print); link integrity preserved.

Findings (current)
- Diagrams and links: Zero issues reported by validators.
- Tests: All green on integration, FSM minimal, synthetic micro.
- Export: Browser‑printable HTML artifact available (docs/export/DCH_TechSpec_v0.1.html).
  - Pandoc CLI path is prepared but requires local installation or running Docker daemon (currently not active).
- Repository hygiene: docs assembled spec present; links normalized for intra‑repo navigation.

Proposed v0.2 edits (document‑level; non‑breaking, clarity and completeness)
1) Formal clarifications
   - Add explicit B‑connectivity definition box with small proof sketch of traversal admissibility.
   - Introduce canonical anchor IDs for all major subsections using {#...} to stabilize cross‑links.

2) Algorithmic pseudocode inserts
   - Provide concise pseudocode blocks for TC‑kNN candidate generation and randomized beam traversal.
   - Provide a worked example for path evidence aggregation and EMA update with numbers.

3) FSM details
   - Specify CMS width/depth defaults and W‑window semantics; add hysteresis on promotions.
   - Clarify canonical label stability requirements under streaming updates.

4) Diagram polish
   - Add figure numbers/titles and cross references; ensure consistent node legends across all Mermaid diagrams.

5) Embedding roadmap
   - Add a “Periodic GraphSAGE refinement” stub with cadence and resource guardrail callouts.

6) Evaluation addenda
   - Add a brief on event‑stream corruption/noise ablations and their expected impact on budgets/pruning.

7) Export appendix
   - Add a short “How to export” appendix with exact pandoc/docker/browser flows.

Code‑adjacent proposals (prototype curvature; tracked for implementation sprints)
- dch_core/embeddings: add placeholder interface for periodic SAGE (disabled by default).
- dch_core/fsm: expose W, s_min, r_min, γ as config and add hysteresis on promotion.
- dch_core/traversal: optional audit trace mode for reviewer reproducibility.

Reviewer actions
- Use the checklist above; record decisions and comments inline under each item (add “Decision”/“Notes” subsections as needed).
- Where “Proposed v0.2 edits” are accepted, mark [x] and leave a short instruction (file + anchor).

Sign‑off rubric (fill)
- Research Lead: decision / name / date
- Engineering Lead: decision / name / date
- Hardware Lead: decision / name / date
- QA Lead: decision / name / date

Appendix — handy commands
- Re‑run diagrams validator:  python3 scripts/check_diagrams.py
- Re‑run link validator:      python3 scripts/check_links.py
- Assemble one‑file spec:     python3 scripts/build_master_md.py
- Open printable artifact:     open docs/export/DCH_TechSpec_v0.1.html

