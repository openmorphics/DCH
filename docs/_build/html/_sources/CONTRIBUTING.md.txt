# Contributing to DCH

Thank you for your interest in contributing to the Dynamic Causal Hypergraph (DCH) project. This document describes how to propose changes, the required development workflow, coding standards, and quality gates that keep the codebase stable and reproducible.

DCH is a research-grade, neuro-symbolic framework. Contributions should preserve:
- Determinism (seeded RNG, reproducible pipelines)
- Torch-free baseline operability
- Clear separation of DCH core logic from backend-specific SNN code
- Interpretability and auditability of learning updates

If you have any questions, please open a GitHub Discussion or Issue.

## Table of Contents
- Getting Started
- Issues and Proposals
- Branching and Commits
- Development Environment
- Code Style, Linting, Typing
- Tests and Coverage
- Documentation
- Optional Dependencies and Gating
- Performance Benchmarks
- Release Checklist
- Code of Conduct

## Getting Started

1) Fork the repository and clone your fork:
   - git clone https://github.com/<your-username>/dch.git

2) Add upstream:
   - git remote add upstream https://github.com/<org>/dch.git

3) Keep your fork in sync:
   - git fetch upstream
   - git checkout main
   - git merge upstream/main

## Issues and Proposals

- Before large changes, please open an Issue describing:
  - Motivation and scope
  - Interfaces affected (core protocols, pipeline)
  - Backward compatibility (default behavior preserved)
  - Test strategy and performance impact

- For small fixes (typos, minor refactors), you may open a PR directly. Link any related Issue.

## Branching and Commits

- Create a feature branch from main:
  - git checkout -b feat/<concise-feature-name>
  - For fixes: fix/<bug>, for docs: docs/<area>

- Use Conventional Commits where practical:
  - feat:, fix:, docs:, test:, chore:, refactor:, perf:, ci:

- Keep commits small and focused. Squash or rebase noisy histories before opening a PR.

## Development Environment

- Python: 3.10–3.12
- OS: Linux/macOS CI-tested
- No GPU required; CPU-only supported across the stack

Recommended steps:
1) Create a virtual environment:
   - python3 -m venv .venv
   - source .venv/bin/activate

2) Install requirements:
   - pip install -r requirements.txt

3) Install pre-commit hooks:
   - pip install pre-commit
   - pre-commit install

4) Run tests:
   - pytest -q

5) Optional containerized workflow:
   - bash scripts/with_docker.sh build cpu
   - bash scripts/with_docker.sh run cpu -- pytest -q

## Code Style, Linting, Typing

- Follow the style and tooling configured in pyproject.toml (formatting, linting, typing).
- Keep functions short and cohesive; favor pure functions in core modules.
- Public functions and classes must include docstrings and type hints.
- Avoid introducing new dependencies without maintainer approval.

## Tests and Coverage

- All new code must include unit tests. Place tests under tests/.
- Keep tests deterministic (seeded RNG; avoid wall-clock/time-based assertions).
- Write fast, torch-free tests by default. Torch-required tests must be explicitly skipped when torch is unavailable.
- For optional dependencies (e.g., tonic, numpy), mock or gate imports to ensure the suite remains green without them.
- Run:
  - pytest -q
  - Consider adding integration or property-based tests where appropriate.

## Documentation

- Update README.md when user-facing behavior changes.
- Update docs/ sections that correspond to your change:
  - Algorithm specs, interfaces, evaluation protocol, hardware appendix, FAQs, etc.
- For new modules, add high-level usage and design rationale to docs/.
- If you add a CLI script, include usage examples in its docstring and reference it from README.md.

## Optional Dependencies and Gating

- Do not import optional dependencies at module top-level. Use lazy import via importlib within function scope and raise a clear ImportError detailing pip/conda install instructions.
- Ensure modules import cleanly even without optional deps installed.
- Tests must verify gating behavior (import succeeds; usage raises friendly error with guidance).

## Performance Benchmarks

- Benchmarks live under benchmarks/ and are CLI-only, stdlib-based.
- They must be deterministic (seeded) and print a single JSON line to stdout.
- Do not gate CI on benchmark results; use them for local regression tracking and ad-hoc CI jobs.

## Release Checklist

- [ ] All tests pass (torch-free and torch-enabled, when applicable)
- [ ] Linting, typing clean
- [ ] Docs updated (README.md, relevant docs/*.md)
- [ ] Changelog (ReleaseNotes) updated
- [ ] Version bumped (pyproject or release tag)
- [ ] CITATION.cff and LICENSE present and accurate

## Code of Conduct

By participating, you agree to abide by our Code of Conduct found at docs/CODE_OF_CONDUCT.md.

Thank you for contributing to DCH!