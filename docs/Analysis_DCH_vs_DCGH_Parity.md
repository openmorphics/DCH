# Analysis: DCH vs DCGH Parity (v0.9.0 → 1.0.0)

Status: Draft – 2025-10-15
Owners: DCH Maintainers

Scope
- This report compares the current DCH prototype implementation to the target DCGH specification across traversal, plasticity, structural rules (DPO), CRC/EAT logging, dual‑proof CMC gating, and DHG construction.
- Source of truth for the spec: [docs/AlgorithmSpecs.md](docs/AlgorithmSpecs.md), [docs/sections/DCH_Section1_FormalFoundations.md](docs/sections/DCH_Section1_FormalFoundations.md), [docs/DCH_TechSpec_v0.1.md](docs/DCH_TechSpec_v0.1.md), [docs/Review_v0.2.md](docs/Review_v0.2.md).
- Code baseline inspected: traversal, DHG, hypergraph, plasticity (EMA + Beta), DPO, CRC/EAT, manifold, embeddings, abstraction, pipeline and tests.

Executive summary
- Parity overall: High for B‑connectivity traversal, EMA plasticity, Beta plasticity option, TC‑kNN DHG, CRC extractor, EAT hash‑chain, DPO prototype and analysis helpers, WL embeddings, abstraction wiring, and pipeline orchestration.
- Major intentional gaps vs DCGH target:
  - Manifold backends: only NoOp present; real feasibility backends not yet implemented; dual‑proof gating wired but default OFF and tested for parity only. See [class ManifoldBackend()](dch_core/manifold.py:32), [class NoOpManifold()](dch_core/manifold.py:79).
  - CRC‑prior influence on traversal scoring is not integrated yet (spec WS‑D). See [class DefaultTraversalEngine()](dch_core/traversal.py:90).
  - DPO audit/replay not implemented (spec WS‑C); current DPO engine is functional and deterministic but not yet audited.
  - GraphSAGE embedding is deferred; WL embedding is implemented and tested. See [class WLHyperpathEmbedding()](dch_core/embeddings/wl.py:57).
- Readiness: Implementation is feature‑complete for v0.9.0 MVP. For 1.0.0, the most material items are (1) at least one non‑trivial manifold backend with tests, (2) CRC‑prior hook in traversal under a feature flag, and (3) DPO audit/replay logging.
- Risk posture: Low to moderate. The current default pathways preserve determinism, use stdlib‑only where possible, and gate advanced features behind flags. The largest compliance risk is running with CMC gating OFF by default.

Key anchor modules (quick index)
- Interfaces: [dch_core/interfaces.py](dch_core/interfaces.py)
- Traversal: [DefaultTraversalEngine.backward_traverse()](dch_core/traversal.py:103)
- DHG TC‑kNN: [DefaultDHGConstructor.generate_candidates_tc_knn()](dch_core/dhg.py:132)
- Plasticity (EMA): [DefaultPlasticityEngine.update_from_evidence()](dch_core/plasticity.py:40)
- Plasticity (Beta): [BetaPlasticityEngine.update_from_evidence()](dch_core/plasticity_beta.py:94), [beta_utils.credible_interval_mc()](dch_core/beta_utils.py:150)
- DPO rules: [DPOEngine.apply()](dch_core/dpo.py:238)
- DPO analysis: [analyze_critical_pairs()](dch_core/dpo_analysis.py:217), [check_termination()](dch_core/dpo_analysis.py:321)
- CRC extractor: [CRCExtractor.make_card()](dch_core/crc.py:68)
- EAT logger: [EATAuditLogger.emit_eat()](dch_pipeline/eat_logger.py:158), [verify_file()](dch_pipeline/eat_logger.py:280)
- Manifold: [ManifoldBackend.check_feasible()](dch_core/manifold.py:61), [NoOpManifold.check_feasible()](dch_core/manifold.py:102)
- Pipeline config: [PipelineConfig](dch_pipeline/pipeline.py:164)

Detailed comparison by component

1) B‑connectivity and backward traversal
Present
- AND‑frontier semantics with deterministic beam search. See [class DefaultTraversalEngine()](dch_core/traversal.py:90), [def backward_traverse()](dch_core/traversal.py:103).
- Temporal gating via [def is_temporally_admissible()](dch_core/interfaces.py:166) and global horizon checks.
- Scoring: multiplicative reliability with length penalty. See [def _length_penalty()](dch_core/traversal.py:41).
- Canonical labels for dedup. See [def _canonical_label()](dch_core/traversal.py:46).
- Tests validate B‑connectivity and admissibility: [tests/test_traversal_credit.py](tests/test_traversal_credit.py).
Missing or partial
- No randomized branching/rng policy; signature reserves rng but uses deterministic top‑K.
- Minimality check is implicit via AND‑frontier and admissibility; explicit minimality proofs not instrumented.
- Beam/DFS/BFS modes limited to beam‑topK; no explicit BFS mode.
Notes
- Complexity matches spec O(D·B·b log(B·b)); see micro‑benchmark scaffold in [benchmarks/benchmark_traversal_complexity.py](benchmarks/benchmark_traversal_complexity.py).

2) Bayesian plasticity (Beta–Bernoulli)
Present
- Posterior mean update with Beta(α0,β0) prior; counters s/f; clamping via state. See [class BetaPlasticityEngine()](dch_core/plasticity_beta.py:59), [def update_from_evidence()](dch_core/plasticity_beta.py:94).
- Utilities: [def posterior_params()](dch_core/beta_utils.py:31), [def posterior_mean()](dch_core/beta_utils.py:58), [def credible_interval_mc()](dch_core/beta_utils.py:150).
- Tests cover positive/negative updates and pipeline selection: [tests/test_plasticity_beta.py](tests/test_plasticity_beta.py).
Partial
- Head‑window timing and edge‑local attribution are present via hyperpath aggregation; path normalization is simple sum‑normalize (as in spec).
- No per‑edge head‑window override beyond delta windows; acceptable for MVP.
Notes
- EMA engine remains default for back‑compat: [class DefaultPlasticityEngine()](dch_core/plasticity.py:31).

3) DPO structural plasticity (GROW/PRUNE/FREEZE)
Present
- Minimal DPO engine and adapter over in‑memory backend. See [class DPOEngine()](dch_core/dpo.py:229), [class DPOGraphAdapter()](dch_core/dpo.py:83).
- Deterministic application and duplicate checks; attribute‑only FREEZE; PRUNE by reliability predicates.
- Critical‑pair and bounded‑termination helpers: [def analyze_critical_pairs()](dch_core/dpo_analysis.py:217), [def check_termination()](dch_core/dpo_analysis.py:321).
- Tests: [tests/test_dpo_rules.py](tests/test_dpo_rules.py), [tests/test_dpo_confluence.py](tests/test_dpo_confluence.py).
Missing or partial
- No audit logger for DPO applications; no replay tool.
- Temporal/guard checks are simplified; relies on DHG invariants when routed through pipeline.

4) CRC/EAT logging frameworks
Present
- CRC extraction with Beta MC composition and credible intervals; deterministic label hashing. See [class CRCExtractor()](dch_core/crc.py:47), [def make_card()](dch_core/crc.py:68).
- EAT JSONL logger with SHA‑256 hash chaining, rotation, and verify. See [class EATAuditLogger()](dch_pipeline/eat_logger.py:79), [def verify_file()](dch_pipeline/eat_logger.py:280).
- Tests validate CRC cards and EAT chain integrity: [tests/test_crc_extractor.py](tests/test_crc_extractor.py), [tests/test_eat_logger.py](tests/test_eat_logger.py).
Partial
- CRC‑prior not yet influencing traversal scoring (planned in ROADMAP WS‑D).
- CRC streaming logger present (crc_logger) but not exercised in smoke by default.

5) CMC dual‑proof validation
Present
- Manifold interface and NoOp backend; gating wired at grow/backward in pipeline. See [class ManifoldBackend()](dch_core/manifold.py:32), [class NoOpManifold()](dch_core/manifold.py:79), [DCHPipeline._dual_proof_enabled()](dch_pipeline/pipeline.py:250).
- Tests confirm default OFF parity and soft/hard semantics when enabled: [tests/test_dual_proof_gating.py](tests/test_dual_proof_gating.py).
Missing
- Real feasibility backend(s) (e.g., LinearReachability); explain() is placeholder in NoOp.
- Operational dashboards/counters beyond per‑step metrics not present.

6) Hypergraph/constructor (TC‑kNN)
Present
- TC‑kNN candidate generation, time clustering, unary and pair/triple enumeration, dedup, per‑head budget heuristic, refractory guard. See [class DefaultDHGConstructor()](dch_core/dhg.py:127), [def generate_candidates_tc_knn()](dch_core/dhg.py:132).
- In‑memory backend with indices, dedup keys, prune, snapshot. See [class InMemoryHypergraph()](dch_core/hypergraph_mem.py:47).
- Tests for unary/pair generation and budgets: [tests/test_dhg_tc_knn.py](tests/test_dhg_tc_knn.py).
Partial
- top_k_recent_spikes uses linear scan; fine for CPU tests; future backends can optimize.

7) Embeddings and abstraction
Present
- WL hyperpath embedding with invariances and reliability toggle; extensive tests. See [class WLHyperpathEmbedding()](dch_core/embeddings/wl.py:57).
- Abstraction engine promoting frequent hyperpaths to HOEs; dedup/idempotent; integrated with FSM. See [DefaultAbstractionEngine.promote()](dch_core/abstraction.py:84).
- Pipeline wiring and FSM thresholds; end‑to‑end test: [tests/test_abstraction_integration.py](tests/test_abstraction_integration.py).
Missing/Deferred
- GraphSAGE periodic embedding (doc‑deferred per ROADMAP).

Implementation gaps and technical debt (repo‑wide scan)
Findings
- Core modules contain very few markers; most TODO/FIXME are in vendored/venv code and ignored here.
- Bare pass after broad exception catches:
  - [dch_core/dpo.py](dch_core/dpo.py) lines 217–218 swallow exceptions with pass in attribute update paths.
  - [dch_core/dpo_analysis.py](dch_core/dpo_analysis.py) lines 69–70 use pass during vertex parsing best‑effort logic.
- Missing features vs spec/ROADMAP:
  - Manifold feasibility backends (WS‑B).
  - DPO audit + replay (WS‑C).
  - CRC‑prior scoring hook (WS‑D).
  - Optional SAGE embedding or doc deferral (WS‑E) – currently deferral documented.
Artifacts
- Machine‑readable extraction written to [artifacts/todos_findings.json](artifacts/todos_findings.json).

Quantitative metrics
Coverage
- .coverage file exists; runtime coverage not regenerated in this pass to keep execution lightweight.
- How to reproduce locally:
  - coverage run -m pytest -q
  - coverage json -o artifacts/coverage.json
  - coverage report
- Summary artifact recorded as [artifacts/coverage_summary.json](artifacts/coverage_summary.json) with status="skipped" and clear instructions.
API surface completeness
- Expected vs implemented (see artifact [artifacts/api_surface_matrix.json](artifacts/api_surface_matrix.json)):
  - TraversalEngine: implemented (backward_traverse).
  - DHGConstructor: implemented (generate_candidates_tc_knn, admit).
  - PlasticityEngine: implemented (EMA); Beta implemented as alternative.
  - DPO: implemented (engine + analysis); audit/replay missing.
  - CRC/EAT: implemented; CRC‑prior in traversal missing.
  - Manifold/CMC: interface implemented; only NoOp backend present.
Performance and complexity
- Traversal micro‑benchmark scaffold: [benchmarks/benchmark_traversal_complexity.py](benchmarks/benchmark_traversal_complexity.py) emits O(K·L·C_in) proxy metrics; expected linear scaling in each factor at small scales.
- Traversal synthetic benchmark: [benchmarks/benchmark_traversal.py](benchmarks/benchmark_traversal.py) – CPU‑only, deterministic.
- Beta plasticity micro‑benchmark: [benchmarks/benchmark_plasticity_beta.py](benchmarks/benchmark_plasticity_beta.py) – updates O(U·M) with U updates over M paths.
Assumptions
- Benchmarks not executed in this pass; results inferred from code and tests.

Risk assessment
- Critical: Lack of real manifold feasibility can violate CMC “dual‑proof” if users enable gating expecting enforcement. Blast radius limited by default OFF and tests that ensure parity.
- High: DPO audit/replay absent – traceability of structural edits is weaker; mitigated by deterministic engine and pending audit WS‑C.
- Medium: CRC‑prior not influencing traversal may reduce sample efficiency; low compliance risk.
- Low: In‑memory backend scaling; acceptable for research/CI, replaceable by optimized backends later.

Implementation effort estimates (calendar days; confidence 60–75%)
- WS‑B LinearReachability manifold backend (CPU, deterministic): M (5–7d). Deps: none or numpy/scipy (extras.stats). Risks: false positives/negatives.
- WS‑C DPO audit logger + replay: S/M (3–5d). Deps: logging JSONL + small CLI. Risks: schema drift; mitigate with tests.
- WS‑D CRC‑prior scoring hook: S (2–3d). Deps: traversal hook and flag; unit/integration test.
- WS‑E GraphSAGE stub or doc deferral: S (1–2d) if doc only; M/L (7–10d) for CPU stub with tests.
- Packaging to extras + CI smoke (already outlined): S/M (3–4d).

Prioritized recommendations aligned to 1.0.0 roadmap
Order (impact/risk reduction vs cost)
1. Implement LinearReachability manifold backend (WS‑B) and keep default OFF; add soft/hard tests.
2. Add DPO audit logger + deterministic replay (WS‑C); wire to pipeline DPO paths.
3. Add CRC‑prior traversal hook (WS‑D) behind flag; preserve default parity; add tests.
4. Keep WL as default; explicitly defer GraphSAGE with docs or add tiny CPU stub (WS‑E).
5. Enable EAT smoke in CI and document verify path (WS‑G).
6. Package heavy deps into extras; ensure core‑only install smoke green (WS‑A).

Call‑path verification map (selected)
- GROW: DHG candidate gen → optional DP:GROW filter → admit or DPO GROW → EAT:GROW. Anchors: [generate_candidates_tc_knn()](dch_core/dhg.py:132), [DCHPipeline.step() grow path](dch_pipeline/pipeline.py:389), [DPOEngine.apply()](dch_core/dpo.py:238), [EATAuditLogger.emit_grow()](dch_pipeline/eat_logger.py:190).
- Backward: traversal → optional DP:BACKWARD hyperpath filter → plasticity update → EAT:UPDATE/EAT paths. Anchors: [backward_traverse()](dch_core/traversal.py:103), [DCHPipeline._check_hyperpath_feasible()](dch_pipeline/pipeline.py:321), [DefaultPlasticityEngine.update_from_evidence()](dch_core/plasticity.py:40), [EATAuditLogger.emit_update()](dch_pipeline/eat_logger.py:209), [EATAuditLogger.emit_eat()](dch_pipeline/eat_logger.py:158).

Appendix A — API surface checklist (spec vs code)
- HypergraphOps: ingest_event, window_query, get_vertex, get_edge, get_incoming_edges, get_outgoing_edges, insert_hyperedges, prune, snapshot. Status: Implemented in [class InMemoryHypergraph()](dch_core/hypergraph_mem.py:47).
- GraphConnectivity: presyn_sources. Status: Implemented in [class StaticGraphConnectivity()](dch_core/hypergraph_mem.py:158).
- DHGConstructor: generate_candidates_tc_knn, admit. Status: Implemented in [class DefaultDHGConstructor()](dch_core/dhg.py:127).
- TraversalEngine: backward_traverse. Status: Implemented in [class DefaultTraversalEngine()](dch_core/traversal.py:90).
- PlasticityEngine: update_from_evidence, prune. Status: Implemented in [class DefaultPlasticityEngine()](dch_core/plasticity.py:31) and [class BetaPlasticityEngine()](dch_core/plasticity_beta.py:59).
- FSMEngine/AbstractionEngine: Implemented and integrated; see [dch_core/fsm.py](dch_core/fsm.py) and [dch_core/abstraction.py](dch_core/abstraction.py).
- ManifoldBackend: interface implemented, NoOp backend present; gating wired in [DCHPipeline.step()](dch_pipeline/pipeline.py:345).
- DPO: Engine + Adapter + Analysis present; audit logging pending.

Appendix B — How to reproduce coverage and quick perf locally
- Coverage:
  - coverage run -m pytest -q
  - coverage json -o artifacts/coverage.json
  - coverage report
- Quick traversal perf (small): python benchmarks/benchmark_traversal_complexity.py --K 4 --L 6 --C_in 8
- Quick plasticity perf (small): python benchmarks/benchmark_plasticity_beta.py

Appendix C — File/link index used in this report
- Spec: [docs/AlgorithmSpecs.md](docs/AlgorithmSpecs.md), [docs/sections/DCH_Section1_FormalFoundations.md](docs/sections/DCH_Section1_FormalFoundations.md), [docs/DCH_TechSpec_v0.1.md](docs/DCH_TechSpec_v0.1.md), [docs/Review_v0.2.md](docs/Review_v0.2.md)
- Roadmap: [docs/ROADMAP.md](docs/ROADMAP.md)
- Core: [dch_core/interfaces.py](dch_core/interfaces.py), [dch_core/dhg.py](dch_core/dhg.py), [dch_core/traversal.py](dch_core/traversal.py), [dch_core/plasticity.py](dch_core/plasticity.py), [dch_core/plasticity_beta.py](dch_core/plasticity_beta.py), [dch_core/beta_utils.py](dch_core/beta_utils.py), [dch_core/dpo.py](dch_core/dpo.py), [dch_core/dpo_analysis.py](dch_core/dpo_analysis.py), [dch_core/crc.py](dch_core/crc.py), [dch_core/manifold.py](dch_core/manifold.py), [dch_core/abstraction.py](dch_core/abstraction.py), [dch_core/embeddings/wl.py](dch_core/embeddings/wl.py)
- Pipeline: [dch_pipeline/pipeline.py](dch_pipeline/pipeline.py), [dch_pipeline/eat_logger.py](dch_pipeline/eat_logger.py), [dch_pipeline/crc_logger.py](dch_pipeline/crc_logger.py), [dch_pipeline/logging_utils.py](dch_pipeline/logging_utils.py), [dch_pipeline/evaluation.py](dch_pipeline/evaluation.py)
- Tests: [tests/test_traversal_credit.py](tests/test_traversal_credit.py), [tests/test_plasticity_beta.py](tests/test_plasticity_beta.py), [tests/test_dhg_tc_knn.py](tests/test_dhg_tc_knn.py), [tests/test_dpo_rules.py](tests/test_dpo_rules.py), [tests/test_dpo_confluence.py](tests/test_dpo_confluence.py), [tests/test_crc_extractor.py](tests/test_crc_extractor.py), [tests/test_eat_logger.py](tests/test_eat_logger.py), [tests/test_dual_proof_gating.py](tests/test_dual_proof_gating.py), [tests/test_abstraction_integration.py](tests/test_abstraction_integration.py), [tests/test_embeddings_wl.py](tests/test_embeddings_wl.py)

End of report