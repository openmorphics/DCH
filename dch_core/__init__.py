# Dynamic Causal Hypergraph (DCH) — Core package
# License: MIT

"""
Core package exposing typed interfaces and default engines for DCH.

Primary modules
- interfaces: typed data models and Protocols for hypergraph structures and engines
- hypergraph_mem: in-memory HypergraphOps and connectivity oracle (CPU-friendly)
- dhg: TC-kNN dynamic hypergraph construction (candidate generation + admission)
- traversal: constrained backward hyperpath traversal (beam with AND-frontier)
- plasticity: evidence-based reliability updates and pruning

This __init__ consolidates common exports for convenience:
    from dch_core import (
        Event, Vertex, Hyperedge, HypergraphSnapshot, Hyperpath, PlasticityState,
        HypergraphOps, GraphConnectivity, DHGConstructor, TraversalEngine, PlasticityEngine,
        InMemoryHypergraph, StaticGraphConnectivity,
        DefaultDHGConstructor, DefaultTraversalEngine, DefaultPlasticityEngine,
    )
"""

from __future__ import annotations

__all__ = [
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
    # Default backends
    "InMemoryHypergraph",
    "StaticGraphConnectivity",
    "DefaultDHGConstructor",
    "DefaultTraversalEngine",
    "DefaultPlasticityEngine",
    # Version
    "__version__",
]

__version__ = "0.1.0"

# Re-export entities and protocols
from .interfaces import (
    Event,
    Vertex,
    Hyperedge,
    HypergraphSnapshot,
    Hyperpath,
    PlasticityState,
    HypergraphOps,
    GraphConnectivity,
    DHGConstructor,
    TraversalEngine,
    PlasticityEngine,
)

# Default in-memory hypergraph backend and connectivity oracle
from .hypergraph_mem import InMemoryHypergraph, StaticGraphConnectivity

# Default engines
from .dhg import DefaultDHGConstructor
from .traversal import DefaultTraversalEngine
from .plasticity import DefaultPlasticityEngine