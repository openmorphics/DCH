# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Backend Abstraction Layer (BAL) — SNN backend-neutral interfaces.

This module defines typed protocols and configuration dataclasses for:
- Encoder: converts event streams into backend-ready spike tensors
- SNNModel: backend model interface (Norse or BindsNET implementations)
- Trainer: training/evaluation runner that does not interfere with DCH credit assignment
- BackendAdapter: factory/provider for encoder, model, and trainer for a given backend

Conventions
- All implementations must be CPU-capable and accept a torch.device to enable CUDA when available.
- Reproducibility: seed control and deterministic flags are handled centrally (pipeline), not in adapters.
- Checkpointing: adapters should implement state_dict/load_state_dict using PyTorch semantics.

References
- Contracts for DCH core structures: dch_core.interfaces
- Decision record and packaging guidance: docs/FrameworkDecision.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

# Optional torch import; interfaces remain usable without torch at runtime
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

# Import shared types (no heavy deps)
from dch_core.interfaces import Event, Window, Timestamp


# -------------------------
# Configuration dataclasses
# -------------------------


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration parameters for event-to-spike encoders."""
    # Temporal binning (microseconds or dataset-native unit)
    time_bin: int = 1000
    # Input spatial geometry (H, W, C) for event cameras when applicable
    shape: Optional[Tuple[int, int, int]] = None
    # Normalization and preprocessing choices
    normalize: bool = True
    # Additional backend-specific kwargs
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration parameters for SNN model construction."""
    input_shape: Optional[Tuple[int, ...]] = None
    hidden_size: Optional[int] = None
    num_classes: Optional[int] = None
    dt: float = 1.0
    threshold: float = 1.0
    backend: str = "norse"  # or "bindsnet"
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration parameters for training/evaluation."""
    max_epochs: int = 10
    max_steps_per_epoch: Optional[int] = None
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = None
    log_interval: int = 50
    device: str = "cpu"
    # Optional AMP/mixed precision control (Norse typically FP32; adapters may ignore)
    use_amp: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)


# -------------------------
# Protocols (interfaces)
# -------------------------


@runtime_checkable
class Encoder(Protocol):
    """
    Event-to-spike encoder.

    Implementations convert a list/stream of Event into a tensor format compatible with the backend.
    Output must include a time-major spike tensor and optional metadata.

    Shapes (typical):
      - spikes: (T, B, *spatial) or (T, B, N) depending on dataset/model
      - meta:   any auxiliary info (e.g., valid lengths per sequence)
    """

    def reset(self) -> None:
        """Reset any internal state (for streaming encoders)."""
        ...

    def encode(
        self,
        events: Sequence[Event],
        window: Window,
        device: Any,  # torch.device when available
    ) -> Tuple[Any, Mapping[str, Any]]:  # torch.Tensor when available
        """
        Convert events within window into spike tensor on device.

        Returns:
            spikes: torch.Tensor (time-major)
            meta:   Mapping with auxiliary information
        """
        ...


@runtime_checkable
class SNNModel(Protocol):
    """
    Backend-neutral SNN model interface.

    Implementations wrap Norse/BindsNET models to provide a unified API surface
    for forward passes and state resets.
    """

    def to(self, device: Any) -> "SNNModel":
        ...

    def train(self, mode: bool = True) -> "SNNModel":
        ...

    def eval(self) -> "SNNModel":
        ...

    def reset_state(self, batch_size: int) -> None:
        """Reset recurrent state between sequences/batches."""
        ...

    def forward(
        self,
        spikes: Any,  # torch.Tensor when available
        state: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Any, Mapping[str, Any]]:
        """
        Run a forward pass.

        Args:
            spikes: time-major spike tensor
            state: optional state dictionary (backend-specific)

        Returns:
            outputs: model outputs (e.g., logits) in a tensor
            new_state: updated state mapping (may be empty)
        """
        ...

    def state_dict(self) -> Mapping[str, Any]:
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        ...


@runtime_checkable
class Trainer(Protocol):
    """
    Backend-neutral trainer interface.

    Must orchestrate training/evaluation without interfering with DCH's separate
    traversal/credit assignment loop.
    """

    def fit(
        self,
        model: SNNModel,
        encoder: Encoder,
        train_loader: Any,
        val_loader: Optional[Any],
        config: TrainerConfig,
    ) -> Mapping[str, Any]:
        """
        Train for config.max_epochs or stopping criterion.

        Returns:
            summary: metrics (e.g., loss/accuracy curves, best epoch)
        """
        ...

    def evaluate(
        self,
        model: SNNModel,
        encoder: Encoder,
        data_loader: Any,
        config: TrainerConfig,
    ) -> Mapping[str, Any]:
        """
        Evaluate model and return metrics mapping.
        """
        ...


@runtime_checkable
class BackendAdapter(Protocol):
    """
    Factory/provider for encoder, model, and trainer for a given backend.
    """

    @property
    def name(self) -> str:
        ...

    def build_encoder(self, config: EncoderConfig) -> Encoder:
        ...

    def build_model(self, config: ModelConfig, device: Any) -> SNNModel:
        ...

    def build_trainer(
        self, config: TrainerConfig, model: SNNModel, encoder: Encoder
    ) -> Trainer:
        ...


__all__ = [
    "EncoderConfig",
    "ModelConfig",
    "TrainerConfig",
    "Encoder",
    "SNNModel",
    "Trainer",
    "BackendAdapter",
]