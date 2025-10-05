# Dynamic Causal Hypergraph (DCH) — Pipeline package
# License: MIT

"""
Pipeline package exposing orchestration utilities for DCH.

Primary exports
- Configs: DHGConfig, TraversalConfig, PlasticityConfig, PipelineConfig
- DCHPipeline: orchestrator that ties together hypergraph storage, DHG constructor,
  traversal, and plasticity engines, with optional encoder for data preparation.

See:
- docs/EVALUATION_PROTOCOL.md
- dch_pipeline/pipeline.py
"""

from __future__ import annotations

from .pipeline import (
    DHGConfig,
    TraversalConfig,
    PlasticityConfig,
    PipelineConfig,
    DCHPipeline,
)

__all__ = [
    "DHGConfig",
    "TraversalConfig",
    "PlasticityConfig",
    "PipelineConfig",
    "DCHPipeline",
]