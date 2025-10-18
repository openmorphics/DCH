# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
CMC feasibility interface (non-enforcing) for Dynamic Causal Hypergraph (DCH).

This module defines a minimal protocol for "manifold" backends that can assess the
feasibility of causal explanations (causes -> effect) within a broader CMC workflow.

Notes
- This interface is non-enforcing in this P2-11 task: it exposes a side-effect free API
  to check feasibility and (optionally) explain it, but the pipeline does not gate on it yet.
- Causes and effect accept permissive types to avoid import cycles with core interfaces.
  The intended usage is:
    * causes: list of Vertex or event-like dicts
    * effect: Vertex or event-like dict
  Backends should not mutate inputs and should be deterministic.

NoOp backend
- A trivial implementation that always returns feasible=True.
- Intended only for plumbing and backward-compatibility validation in this task.

Future work
- P2-12 will introduce dual-proof gating using this interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ManifoldBackend(Protocol):
    """
    Minimal feasibility API for CMC-style manifold backends.

    Requirements
    - Side-effect free checks (no logging or I/O here).
    - Deterministic given the same inputs.

    Methods
    - name(): string identifier of the backend (e.g., 'noop').
    - version(): semantic or simple version string (e.g., '1.0').
    - serialize_config(): a JSON-serializable dict describing backend configuration.
    - check_feasible(causes, effect, context): return True/False feasibility.
      * causes: list of Vertex or event-like dicts (typed as Any to avoid import cycles)
      * effect: Vertex or event-like dict (typed as Any)
      * context: optional mapping with additional hints; must be read-only from backend POV.
    - explain(causes, effect, context): optional structured explanation dict.
      The default behavior may return an empty dict when not applicable.
    """

    def name(self) -> str:
        ...

    def version(self) -> str:
        ...

    def serialize_config(self) -> Dict[str, Any]:
        ...

    def check_feasible(
        self,
        causes: List[Any],
        effect: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        ...

    # Optional: backends may provide a structured explanation of the feasibility decision.
    def explain(
        self,
        causes: List[Any],
        effect: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ...


class NoOpManifold(ManifoldBackend):
    """
    Trivial manifold backend that declares all inputs feasible.

    Behavior
    - name() = 'noop'
    - version() = '1.0'
    - serialize_config() = {'type': 'noop'}
    - check_feasible(...) -> True
    - explain(...) -> {'type': 'noop', 'feasible': True}

    This backend is intended for wiring validation and backward compatibility tests only.
    """

    def name(self) -> str:
        return "noop"

    def version(self) -> str:
        return "1.0"

    def serialize_config(self) -> Dict[str, Any]:
        return {"type": "noop"}

    def check_feasible(
        self,
        causes: List[Any],
        effect: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return True

    def explain(
        self,
        causes: List[Any],
        effect: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {"type": "noop", "feasible": True}


__all__ = ["ManifoldBackend", "NoOpManifold"]