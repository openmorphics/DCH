# Supplementary Materials

This page tracks supporting figures, algorithmic details, and extended derivations. Place final artifacts under `docs/figures/` and export assembled bundles to `docs/export/`.

Note: These materials are optional for using the software. They support paper/report packaging and are maintained alongside the docsite.

---

## Figures (placeholders)

Store figure sources in `docs/figures/` (prefer vector formats: `.pdf`, `.svg`, or high-res `.png`), and reference them from specification docs when finalized.

Suggested placeholders:
- Figure S1 — DCH end-to-end flow: ingest → DHG → traversal → plasticity → FSM → abstraction.
- Figure S2 — Temporal windowing and TC‑kNN candidate selection.
- Figure S3 — B-connected backward traversal beam search with AND-frontier.
- Figure S4 — Streaming frequent hyperpath mining with decay and hysteresis.
- Figure S5 — Promotion to higher-order hyperedges (HOEs) and dedup guards.
- Figure S6 — Task-aware scaffolding (FREEZE/REUSE/ISOLATE) decision surfaces.

Add the final files under `docs/figures/` and update links in the relevant docs (e.g., `docs/AlgorithmSpecs.md`, `docs/EVALUATION_PROTOCOL.md`).

---

## Algorithms

Reference canonical algorithm descriptions (to be expanded with step-by-step pseudocode and complexity notes):

- DHG candidate construction (TC‑kNN)
  - Core implementation: `dch_core/dhg.py`
  - Outline: sliding time window; candidate enumeration under δ_causal; refractory guard; dedup; per-head budgets.

- Backward traversal with B-connectivity
  - Core implementation: `dch_core/traversal.py`
  - Outline: beam search; AND-frontier; temporal admissibility; reliability-composed scoring; canonical labeling.

- Plasticity and pruning
  - Core implementation: `dch_core/plasticity.py`
  - Outline: evidence aggregation over hyperpaths; EMA update with clamping; decay; prune sweep.

- Streaming FSM
  - Core implementation: `dch_core/fsm.py`
  - Outline: decayed label counters; hysteresis; promotion to “frequent”.

- Abstraction (HOE promotion)
  - Core implementation: `dch_core/abstraction.py`
  - Outline: promotion criteria; acyclicity and dedup guards; label composition.

- Scaffolding policy
  - Core implementation: `dch_core/scaffolding.py`
  - Outline: similarity thresholds (REUSE vs ISOLATE); FREEZE strategies; resistance scaling.

For narrative specifications and equations, start from:
- `docs/AlgorithmSpecs.md`
- `docs/sections/` (formal foundations, complexity/resource, FSM, abstraction, scaffolding)

---

## Extended proofs and derivations

Provide formal notes and derivations that complement `docs/AlgorithmSpecs.md`:
- Temporal admissibility bounds and refractory conditions
- Reliability composition properties and clamping stability
- FSM decay/hysteresis stability under noise processes
- Acyclicity guarantees for HOE promotion
- Complexity envelopes (beam/combination budgets; memory footprints)

Place working notes in `docs/sections/` and link finalized derivations here.

---

## Archival and exports

- Export final PDFs/HTML to `docs/export/` for archival and distribution.
- See `docs/ReleaseNotes.md` for tagging guidance and a recommended list of exported artifacts.
- Build helper scripts (optional): `scripts/export_pdf.sh`, `scripts/export_html.py`.
