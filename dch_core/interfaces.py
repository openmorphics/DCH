# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Dynamic Causal Hypergraph (DCH) — Core typed interfaces and data models.

This module defines:
- Type aliases for identifiers and time
- Data models: Event, Vertex, Hyperedge, HypergraphSnapshot, Hyperpath, PlasticityState
- Protocols (interfaces) for core engines:
    * HypergraphOps (storage + indices)
    * GraphConnectivity (presynaptic source queries)
    * DHGConstructor (TC-kNN candidate generation and admission)
    * TraversalEngine (constrained backward hyperpath traversal)
    * PlasticityEngine (evidence-based reliability updates + pruning)
    * EmbeddingEngine (hyperpath embeddings; e.g., WL-style)
    * FSMEngine (streaming frequent hyperpath mining)
    * AbstractionEngine (promotion to higher-order edges)
    * ScaffoldingPolicy (task-aware FREEZE/PRUNE/GROW policy)
- Determinism/seed config helper structures

References:
- docs/Interfaces.md
- docs/FrameworkDecision.md
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    runtime_checkable,
    NewType,
)

# ---------- Type aliases and identifiers ----------

Timestamp = int  # microseconds or dataset-native time unit
NeuronId = int

VertexId = NewType("VertexId", str)  # canonical string id, e.g., "n@t"
EdgeId = NewType("EdgeId", str)
ReliabilityScore = float  # in [0.0, 1.0]
Window = Tuple[Timestamp, Timestamp]  # closed interval [t0, t1], t0 <= t1


def make_vertex_id(neuron_id: NeuronId, t: Timestamp) -> VertexId:
    """Create a canonical vertex id as 'neuron@time'."""
    return VertexId(f"{neuron_id}@{t}")


def make_edge_id(head: VertexId, tail: Iterable[VertexId], t: Timestamp) -> EdgeId:
    """Create a canonical edge id based on sorted tail, head, and a timestamp nonce."""
    tail_sorted = ",".join(sorted(tail))
    return EdgeId(f"{head}&{tail_sorted}#{t}")


# ---------- Entities ----------


@dataclass(frozen=True)
class Event:
    neuron_id: NeuronId
    t: Timestamp
    meta: Optional[Mapping[str, Any]] = None

    def to_jsonable(self) -> Dict[str, Any]:
        return {"neuron_id": self.neuron_id, "t": self.t, "meta": dict(self.meta or {})}


@dataclass(frozen=True)
class Vertex:
    id: VertexId
    neuron_id: NeuronId
    t: Timestamp


@dataclass
class Hyperedge:
    id: EdgeId
    tail: Set[VertexId]  # non-empty
    head: VertexId
    # Temporal constraints for admissibility (Allen-style, simplified)
    delta_min: int  # minimum presyn->post delay (inclusive)
    delta_max: int  # maximum presyn->post delay (inclusive)
    # Refractory constraint for the postsynaptic neuron (minimum spacing between head events)
    refractory_rho: int

    # Learnable/maintained fields
    reliability: ReliabilityScore = 0.10
    counts_success: int = 0
    counts_miss: int = 0
    last_update_t: Optional[Timestamp] = None

    # Optional provenance, budgets, and metadata
    provenance: Optional[str] = None
    budget_class: Optional[str] = None  # for candidate budgets
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tail:
            raise ValueError("Hyperedge.tail must be non-empty")
        if self.head in self.tail:
            raise ValueError("Hyperedge.head cannot be in tail")
        if self.delta_min > self.delta_max:
            raise ValueError("delta_min must be <= delta_max")
        if not (0.0 <= self.reliability <= 1.0):
            raise ValueError("reliability must be within [0.0, 1.0]")

    def to_jsonable(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tail"] = sorted(list(self.tail))
        return d


@dataclass
class HypergraphSnapshot:
    """Immutable snapshot for persistence/export/analysis."""
    vertices: Mapping[VertexId, Vertex]
    hyperedges: Mapping[EdgeId, Hyperedge]
    incoming: Mapping[VertexId, Set[EdgeId]]
    outgoing: Mapping[VertexId, Set[EdgeId]]
    # Optional time indices or summaries may be provided for diagnostics
    summary: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class Hyperpath:
    """DAG-like record of a validated causal chain satisfying B-connectivity and temporal constraints."""
    head: VertexId
    edges: Tuple[EdgeId, ...]
    score: float  # product/min of reliabilities (with optional penalties)
    length: int
    label: Optional[str] = None  # canonical label for FSM

    def to_jsonable(self) -> Dict[str, Any]:
        return {"head": self.head, "edges": list(self.edges), "score": self.score, "length": self.length, "label": self.label}


@dataclass
class PlasticityState:
    ema_alpha: float = 0.10  # exponential moving average factor
    reliability_clamp: Tuple[float, float] = (0.02, 0.98)
    decay_lambda: float = 0.0  # optional time decay applied periodically
    freeze: bool = False
    prune_threshold: float = 0.05

    def clamp(self, r: float) -> float:
        lo, hi = self.reliability_clamp
        return max(lo, min(hi, r))


# ---------- Constraints helpers ----------


def is_temporally_admissible(
    tail_times: Sequence[Timestamp],
    head_time: Timestamp,
    delta_min: int,
    delta_max: int,
) -> bool:
    """Check if every presyn timestamp satisfies head_time - delta_max <= t_i <= head_time - delta_min."""
    lower = head_time - delta_max
    upper = head_time - delta_min
    return all((lower <= ti <= upper) for ti in tail_times)


# ---------- Protocols (interfaces) ----------


@runtime_checkable
class HypergraphOps(Protocol):
    """Storage and indices for the evolving hypergraph V(t), E(t)."""

    def ingest_event(self, event: Event) -> Vertex:
        """Materialize an event as a Vertex, update time and neuron indices, and return the Vertex."""
        ...

    def window_query(self, window: Window) -> Sequence[Vertex]:
        """Fetch vertices with t in [t0, t1]."""
        ...

    def get_vertex(self, vid: VertexId) -> Optional[Vertex]:
        ...

    def get_edge(self, eid: EdgeId) -> Optional[Hyperedge]:
        ...

    def get_incoming_edges(self, vid: VertexId) -> Set[EdgeId]:
        ...

    def get_outgoing_edges(self, vid: VertexId) -> Set[EdgeId]:
        ...

    def insert_hyperedges(self, candidates: Sequence[Hyperedge]) -> Sequence[EdgeId]:
        """Admit candidates with dedup, budgets, and index updates; return admitted edge ids."""
        ...

    def prune(self, now_t: Timestamp, prune_threshold: float) -> int:
        """Remove edges below threshold and return count pruned."""
        ...

    def snapshot(self) -> HypergraphSnapshot:
        ...


@runtime_checkable
class GraphConnectivity(Protocol):
    """Minimal connectivity oracle needed by TC-kNN candidate generation."""

    def presyn_sources(self, neuron_id: NeuronId) -> Iterable[NeuronId]:
        """Return neuron ids that have synaptic connectivity to the given postsynaptic neuron."""
        ...


@runtime_checkable
class DHGConstructor(Protocol):
    """Dynamic Hypergraph Construction (TC-kNN + admission)."""

    def generate_candidates_tc_knn(
        self,
        hypergraph: HypergraphOps,
        connectivity: GraphConnectivity,
        head_vertex: Vertex,
        window: Window,
        k: int,
        combination_order_max: int,
        causal_coincidence_delta: int,
        budget_per_head: int,
        init_reliability: float,
        refractory_rho: int,
    ) -> Sequence[Hyperedge]:
        """
        Generate candidate hyperedges for a head vertex by:
        - searching per-presyn recent spikes in window (temporal TC-kNN)
        - forming unary and higher-order combinations within causal_coincidence_delta
        - returning up to budget_per_head candidates with initial reliability
        """
        ...

    def admit(
        self, hypergraph: HypergraphOps, candidates: Sequence[Hyperedge]
    ) -> Sequence[EdgeId]:
        """Insert candidates into the hypergraph with dedup and budgets; return admitted ids."""
        ...


@runtime_checkable
class TraversalEngine(Protocol):
    """Constrained backward hyperpath traversal with B-connectivity and temporal logic."""

    def backward_traverse(
        self,
        hypergraph: HypergraphOps,
        target: Vertex,
        horizon: int,
        beam_size: int,
        rng: Optional[Callable[[int], float]],
        refractory_enforce: bool = True,
    ) -> Sequence[Hyperpath]:
        """
        Explore admissible hyperpaths ending at target within a horizon, respecting:
        - B-connectivity: all tail vertices must be present to traverse an edge
        - Temporal constraints: delta_min/max windows
        - Refractory: spacing between heads with same postsyn neuron
        Returns a set of hyperpaths scored by reliability composition.
        """
        ...


@runtime_checkable
class PlasticityEngine(Protocol):
    """Evidence-based reliability updates and pruning ('predict and confirm/miss')."""

    def update_from_evidence(
        self,
        hypergraph: HypergraphOps,
        hyperpaths: Sequence[Hyperpath],
        sign: int,  # +1 for reward/confirm, -1 for error/depress
        now_t: Timestamp,
        state: PlasticityState,
    ) -> Mapping[EdgeId, ReliabilityScore]:
        """
        Aggregate hyperpath evidence to update edge reliabilities with EMA and clamping.
        Return the updated reliabilities mapping.
        """
        ...

    def prune(
        self,
        hypergraph: HypergraphOps,
        now_t: Timestamp,
        state: PlasticityState,
    ) -> int:
        """Prune edges below threshold; return count pruned."""
        ...


@runtime_checkable
class EmbeddingEngine(Protocol):
    """Online hyperpath embedding (e.g., WL-style, causally context-aware)."""

    def embed(self, hyperpath: Hyperpath) -> Sequence[float]:
        ...

    def batch_embed(self, hyperpaths: Sequence[Hyperpath]) -> Sequence[Sequence[float]]:
        ...


@runtime_checkable
class FSMEngine(Protocol):
    """Streaming Frequent Subgraph (Hyperpath) Mining."""

    def observe(
        self,
        hyperpaths: Sequence[Hyperpath],
        now_t: Timestamp,
    ) -> Sequence[str]:
        """
        Observe hyperpaths, update counts, and return labels crossing frequency threshold for promotion.
        """
        ...


@runtime_checkable
class AbstractionEngine(Protocol):
    """Promotion of frequent hyperpaths into higher-order hyperedges (HOEs)."""

    def promote(
        self,
        hypergraph: HypergraphOps,
        label: str,
        guard_acyclic: bool = True,
    ) -> Optional[EdgeId]:
        """
        Create a new higher-order hyperedge that abstracts a reliable chain,
        preserving provenance and preventing cycles/duplication.
        """
        ...


@runtime_checkable
class ScaffoldingPolicy(Protocol):
    """Task-aware FREEZE/PRUNE/GROW policy based on similarity and governance signals."""

    def evaluate_task_similarity(self, activations_signature: Mapping[str, float]) -> float:
        """Return similarity score [0,1] relative to prior tasks."""
        ...

    def plan(
        self,
        similarity: float,
        state: PlasticityState,
    ) -> Mapping[str, Any]:
        """
        Return a policy dict guiding:
        - freeze sets (edge ids to lock)
        - growth bias regions
        - pruning aggressiveness
        """
        ...


# ---------- Determinism and environment ----------


@dataclass(frozen=True)
class SeedConfig:
    python: int = 0
    numpy: int = 0
    torch: int = 0
    extra: Mapping[str, int] = field(default_factory=dict)


@dataclass
class EnvironmentFingerprint:
    python_version: str
    platform: str
    torch_version: Optional[str] = None
    cuda: Optional[str] = None
    cudnn: Optional[str] = None
    packages: Mapping[str, str] = field(default_factory=dict)


__all__ = [
    # Types
    "Timestamp",
    "NeuronId",
    "VertexId",
    "EdgeId",
    "ReliabilityScore",
    "Window",
    # Entities
    "Event",
    "Vertex",
    "Hyperedge",
    "HypergraphSnapshot",
    "Hyperpath",
    "PlasticityState",
    # Protocols
    "HypergraphOps",
    "GraphConnectivity",
    "DHGConstructor",
    "TraversalEngine",
    "PlasticityEngine",
    "EmbeddingEngine",
    "FSMEngine",
    "AbstractionEngine",
    "ScaffoldingPolicy",
    # Determinism
    "SeedConfig",
    "EnvironmentFingerprint",
    # Helpers
    "make_vertex_id",
    "make_edge_id",
    "is_temporally_admissible",
]