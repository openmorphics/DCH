# DCGH Comprehensive Task Breakdown and Implementation Roadmap

Version: 0.9.0 (MVP) • Target: 1.0.0

Objectives
- Deliver a transparent, auditable DCGH core with minimal install surface.
- Align implementation with the formal spec (B‑connectivity, backward traversal, Bayesian plasticity, DPO structural plasticity, CRC/EAT logs, CMC dual‑proof).
- Keep advanced features opt‑in via extras and feature flags; preserve deterministic CI verification.

Architecture anchors (code references)
- Core contracts and invariants: [dch_core/interfaces.py](dch_core/interfaces.py)
  - [def is_temporally_admissible()](dch_core/interfaces.py:166)
  - [class HypergraphOps()](dch_core/interfaces.py:182), [class TraversalEngine()](dch_core/interfaces.py:259), [class PlasticityEngine()](dch_core/interfaces.py:282)
  - [class Event()](dch_core/interfaces.py:72), [class Vertex()](dch_core/interfaces.py:82), [class Hyperedge()](dch_core/interfaces.py:89), [class Hyperpath()](dch_core/interfaces.py:138), [class PlasticityState()](dch_core/interfaces.py:151)
- In‑memory hypergraph: [class InMemoryHypergraph()](dch_core/hypergraph_mem.py:47), [class StaticGraphConnectivity()](dch_core/hypergraph_mem.py:158)
- TC‑kNN construction: [class DefaultDHGConstructor()](dch_core/dhg.py:127), [def generate_candidates_tc_knn()](dch_core/dhg.py:132), [def admit()](dch_core/dhg.py:258)
  - Helpers: [def _top_k_recent_spikes()](dch_core/dhg.py:41), [def _cluster_by_time_proximity()](dch_core/dhg.py:57), [def _refractory_ok()](dch_core/dhg.py:79), [def _dedup_by_key()](dch_core/dhg.py:99), [def _score_candidate()](dch_core/dhg.py:114)
- Traversal (B‑connectivity): [class DefaultTraversalEngine()](dch_core/traversal.py:90), [def backward_traverse()](dch_core/traversal.py:103)
- Plasticity: [class DefaultPlasticityEngine()](dch_core/plasticity.py:31), [def update_from_evidence()](dch_core/plasticity.py:40), [def prune()](dch_core/plasticity.py:90)
  - Beta: [class BetaPlasticityEngine()](dch_core/plasticity_beta.py:59), [def update_from_evidence()](dch_core/plasticity_beta.py:94), [def prune()](dch_core/plasticity_beta.py:168)
  - Beta utils: [def posterior_params()](dch_core/beta_utils.py:31), [def posterior_mean()](dch_core/beta_utils.py:58), [def credible_interval_mc()](dch_core/beta_utils.py:150)
- Structural plasticity (DPO): [class DPOEngine()](dch_core/dpo.py:229), [def apply()](dch_core/dpo.py:238)
- Symbolic layer: [class StreamingFSMEngine()](dch_core/fsm.py:47), [class DefaultAbstractionEngine()](dch_core/abstraction.py:64), [def promote()](dch_core/abstraction.py:84), [class CRCExtractor()](dch_core/crc.py:47)
- CMC dual‑proof: [class ManifoldBackend()](dch_core/manifold.py:32), [def check_feasible()](dch_core/manifold.py:61), [class NoOpManifold()](dch_core/manifold.py:79), [def check_feasible()](dch_core/manifold.py:102)
- Pipeline & config: [class PipelineConfig()](dch_pipeline/pipeline.py:164), [class ManifoldConfig()](dch_pipeline/pipeline.py:143), [class DualProofConfig()](dch_pipeline/pipeline.py:150), [class DCHPipeline()](dch_pipeline/pipeline.py:194)
- Evaluation: [def run_quick_synthetic()](dch_pipeline/evaluation.py:547), [def run_quick_dataset()](dch_pipeline/evaluation.py:697)
- EAT logger: [dch_pipeline/eat_logger.py](dch_pipeline/eat_logger.py), [def verify_file()](dch_pipeline/eat_logger.py:280)
- CLIs: [scripts/run_quick_experiment.py](scripts/run_quick_experiment.py:1) → [def run_main()](scripts/run_quick_experiment.py:36)
- CI packaging: [.github/workflows/ci.yml](.github/workflows/ci.yml:64), [pyproject.toml](pyproject.toml:1)

Executive summary
- Implemented: typed contracts; in‑memory hypergraph; TC‑kNN with temporal guards; constrained backward traversal; EMA and Beta plasticity; DPO prototype; CRC/EAT logging; deterministic quick runners; CI packaging/release.
- Gaps: heavy mandatory deps; manifold backends limited to NoOp; GraphSAGE missing (WL present); Hydra config flow partial; dataset downloader/protocol wiring incomplete; ensure EAT/DPO audit are exercised in smoke; local 3.9 install blocked by Python ≥3.10 (by design, CI covers 3.10–3.12).
- Priorities: (1) refactor deps into extras; (2) implement 1+ manifold backend with tests; (3) add GraphSAGE stub or defer in docs; (4) finalize dataset/Hydra plumbing; (5) keep install/CLI verification in CI for Python ≥3.10; (6) extend DPO audit + EAT smoke.

Milestones and timeline (indicative)
- M0 (DONE): MVP 0.9.0 packaging, CI gates, quick runners, docs.
- M1 (1–2 wks): Packaging → extras; core‑only install smoke in CI; docs update.
- M2 (1–2 wks): CMC dual‑proof backend(s) + tests; default OFF.
- M3 (1 wk): DPO audit events + deterministic replay; EAT smoke on.
- M4 (1–2 wks): CRC priors in traversal; WL OK; GraphSAGE stub or doc deferral; tests.
- M5 (1–2 wks): Dataset downloader + optional Hydra CLI; micro configs remain CI default.
- M6 (1 wk): Documentation polish (install matrix, EAT verify, dual‑proof how‑to).
- M7 (1 wk): Release 1.0.0 (tag, checksums, TestPyPI/PyPI).

Workstreams, tasks, and acceptance criteria

WS‑A Packaging and dependency footprint (M1)
- A1: Move heavy deps to extras in [pyproject.toml](pyproject.toml:1).
  - Core: numpy, networkx, pyyaml, typing_extensions, tqdm, rich.
  - extras.snn: torch, norse; extras.datasets: tonic, pandas; extras.stats: scipy, statsmodels, tensorboard; extras.baselines: scikit‑learn (+bindsnet/torchvision if used); extras.docs: sphinx, myst‑parser, furo.
- A2: Guard optional imports in CLIs [scripts.run_quick_experiment.run_main()](scripts/run_quick_experiment.py:36), [scripts.run_dataset_experiment.run_main()](scripts/run_dataset_experiment.py:40); print clear guidance when extras missing.
- A3: CI “install‑wheel‑smoke”: install core‑only + NumPy, run dch‑quick synthetic, upload artifacts [.github/workflows/ci.yml](.github/workflows/ci.yml:64).
- A4: Update install docs with Python ≥3.10 and extras: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md:1).
- Acceptance: Core wheel installs; dch‑quick produces metrics.jsonl; docs reflect extras and Python ≥3.10.

WS‑B Manifold feasibility backends and dual‑proof (M2)
- B1: Implement LinearReachability backend (new file, e.g., dch_core/manifold_linear.py) implementing [ManifoldBackend.check_feasible()](dch_core/manifold.py:61).
- B2: (Optional) ConvexHull backend using SciPy (extras.stats), feature‑gated.
- B3: Wire backend selection via [PipelineConfig.manifold](dch_pipeline/pipeline.py:177) and gating via [PipelineConfig.dual_proof](dch_pipeline/pipeline.py:180); default OFF.
- B4: Tests: unit/property tests for feasibility; traversal/grow gating (soft/hard) deterministic.
- B5: Docs: dual‑proof guide with examples and toggles.
- Acceptance: Backend tests pass; dual‑proof gating tests pass; default OFF parity maintained.

WS‑C DPO audit logging and replay (M3)
- C1: Add DPO audit logger (dch_pipeline/dpo_logger.py) to record [DPOEngine.apply()](dch_core/dpo.py:238) outcomes (rule, params, pre/post, thresholds, ts).
- C2: Integrate with pipeline DPO hook points (no behavior changes).
- C3: Add replay tool (or extend existing) to reapply DPO events deterministically.
- C4: Tests: schema validation, presence in audit, replay equivalence.
- Acceptance: DPO events emitted and replay reconstructs final state.

WS‑D Symbolic priors in traversal (CRC influence) (M4)
- D1: Add optional CRC‑prior term in traversal scoring under flag; wire into [DefaultTraversalEngine.backward_traverse()](dch_core/traversal.py:103) via a small scoring hook.
- D2: Provide CRC→prior mapping from [CRCExtractor.make_card()](dch_core/crc.py:68) motifs to scoring.
- D3: Tests: integration where CRC increases intended path probability and affects metrics.
- Acceptance: Effect measurable under flag; default OFF parity preserved.

WS‑E Embeddings (WL present; SAGE stub or doc) (M4)
- E1: Add [embeddings/sage.py](dch_core/embeddings/wl.py:1) stub (CPU mean aggregator) gated by extras.snn; or
- E2: Update docs to defer SAGE and clarify WL is default.
- Acceptance: No doc drift; CI unaffected by optional SAGE.

WS‑F Dataset downloader and Hydra CLI (optional) (M5)
- F1: Implement [scripts/download_datasets.py](scripts/download_datasets.py:1) with SHA256 and manifest logging; avoid downloads in CI.
- F2: Optional Hydra entrypoint (scripts/main_hydra.py) with config groups; keep quick CLIs intact.
- F3: Ensure “micro” configs (e.g., [configs/micro.yaml](configs/micro.yaml:1)) remain offline‑safe defaults for CI.
- Acceptance: Manual dataset runs OK; CI stays offline with micro configs.

WS‑G EAT emission and integrity smoke (M3–M4)
- G1: Expose audit_log_path in quick path via [class PipelineConfig()](dch_pipeline/pipeline.py:164) to emit EAT in smoke.
- G2: Test verifies non‑empty EAT and hash‑chain using [eat_logger.verify_file()](dch_pipeline/eat_logger.py:280).
- Acceptance: EAT artifact emitted and verified deterministically.

WS‑H CI, release automation, distribution (ongoing)
- H1: Keep matrix (3.10–3.12; linux/macos) tests lint/type/test jobs.
- H2: Ensure dist build/upload artifact; install‑wheel‑smoke runs and uploads quick artifacts [.github/workflows/ci.yml](.github/workflows/ci.yml:64).
- H3: Optional TestPyPI publish (workflow_dispatch + secrets); install‑from‑TestPyPI smoke.
- Acceptance: CI green; artifacts uploaded; release draft maintained.

WS‑I Performance and stability (post‑M5)
- I1: Benchmarks (traversal/plasticity) with soft SLOs for quick synthetic; track trend in CI artifacts.
- I2: Memory checks around candidate budgets and prune sweeps.
- Acceptance: Stable latency and memory profiles on microbenchmarks.

WS‑J Documentation and examples (cross‑cutting)
- J1: Update [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md:1) for Python ≥3.10 and extras.
- J2: Add examples: EAT verification; dual‑proof toggles; CRC‑prior demo.
- J3: Align formal sections; clearly mark deferred items (e.g., SAGE).
- Acceptance: Docs build clean; examples runnable; no drift.

Dependencies and sequencing
- M1 (WS‑A) precedes all installability work in CI.
- M2 (WS‑B) uses existing pipeline hooks; default OFF to avoid regressions.
- M3 (WS‑C,G) independent; adds audit coverage and smoke verification.
- M4 (WS‑D,E) builds on CRCs/WL; SAGE stub optional or deferred in docs.
- M5 (WS‑F) independent; ensure CI remains offline via micro configs.

Risks and mitigations
- Install friction from heavy deps → extras‑only core; guard imports; friendly CLI errors.
- Dual‑proof regressions → default OFF; strong tests; feature flags.
- Documentation drift → doc PRs tied to feature branches; defer clearly when needed.
- Reproducibility of audits → EAT hash‑chain; DPO append‑only audit with replay.

Release gates
- Gate 1 (M1): Core‑only wheel install passes in CI; docs updated for extras and Python ≥3.10.
- Gate 2 (M2–M3): Manifold backend tests + dual‑proof gating + DPO audit logging + EAT smoke.
- Gate 3 (M4–M5): CRC‑prior demo test; dataset downloader/Hydra optional; docs/examples complete.
- Gate 4 (M7): Tag v1.0.0; publish artifacts (sdist/wheel, checksums); optional TestPyPI→PyPI promotion.

Checklists (working)
- [ ] Refactor deps to extras; guard CLI imports; CI core install‑smoke green.
- [ ] Implement LinearReachability manifold backend; tests; docs; default OFF.
- [ ] Add DPO audit logger + replay tool; tests; example.
- [ ] Enable EAT emission in smoke; verify chain integrity.
- [ ] Add CRC‑prior scoring hook + integration test.
- [ ] WL present; GraphSAGE stub (optional) or doc deferral; CI unaffected.
- [ ] Implement dataset downloader; optional Hydra entrypoint; micro configs default in CI.
- [ ] Update docs (install matrix, extras, EAT verification, dual‑proof how‑to).
- [ ] Prepare 1.0.0 release artifacts, checksums, and publish plan.

Acceptance references (spec → code)
- B‑connectivity and traversal: [DefaultTraversalEngine.backward_traverse()](dch_core/traversal.py:103)
- Bayesian plasticity (Beta): [BetaPlasticityEngine.update_from_evidence()](dch_core/plasticity_beta.py:94), [credible_interval_mc()](dch_core/beta_utils.py:150)
- Structural plasticity via DPO: [DPOEngine.apply()](dch_core/dpo.py:238)
- CRC extraction: [CRCExtractor.make_card()](dch_core/crc.py:68)
- EAT integrity: [eat_logger.verify_file()](dch_pipeline/eat_logger.py:280)
- Dual‑proof toggles: [PipelineConfig.manifold](dch_pipeline/pipeline.py:177), [PipelineConfig.dual_proof](dch_pipeline/pipeline.py:180)
- Packaging/CI: [pyproject.toml](pyproject.toml:1), [.github/workflows/ci.yml](.github/workflows/ci.yml:64)

Versioning and release notes
- Current version: [__version__](dch_core/__init__.py:49) = "0.9.0"
- Next: 1.0.0 upon completing Gate 4; publish sdist/wheel with checksums; update docs/export index.

Ownership and review cadence
- Assign an owner per WS; weekly milestone review; CI gate report attached to PRs; release checklist maintained in docs.

Appendix — quick commands (reference)
- Build dists: `python -m build` (requires setuptools; see [pyproject.toml](pyproject.toml:1))
- Quick synthetic (module): `python -m scripts.run_quick_experiment --mode synthetic --artifacts-dir artifacts/quick`
- Verify EAT hash‑chain: `python - < 'from dch_pipeline.eat_logger import verify_file; print(verify_file("artifacts/quick/eat.jsonl"))'`
- CI install‑wheel smoke (see [.github/workflows/ci.yml](.github/workflows/ci.yml:64))

End of roadmap