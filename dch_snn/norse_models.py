# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Norse-based SNN model wrappers (import-safe, torch-optional by default).

This module provides:
- create_model(config): factory returning a torch.nn.Module (Norse LIF MLP) and metadata.
- prepare_input(spikes, device, dtype): thin adapter to map encoder outputs to tensors.

Design goals
- Import-safe: no top-level imports of torch or norse.
- Lazy imports: dependencies are imported when create_model/prepare_input are called.
- Deterministic-friendly: no seeding here; pipeline/runner controls seeding.
- Minimal surface: small LIF-based MLP supporting unrolled temporal forward over inputs shaped (T, B, N)
  or (T, 1, N) to match SimpleBinnerEncoders in dch_data.encoders.

Expected config dictionary layout (Hydra-style or plain YAML):
model:
  name: "norse_lif"
  unroll: true
lif:
  # neuron parameters (optional; defaults are used if unspecified)
  tau_mem: 20.0
  tau_syn: 5.0
  v_th: 1.0
  v_reset: 0.0
  refractory: 2.0
topology:
  input_size: 3
  hidden_sizes: [128]
  num_classes: 3
  dropout: 0.0

Usage
- At runtime, the CLI should compute the encoder meta["N"] and override topology.input_size accordingly.
- See [create_model()](dch_snn/norse_models.py:79).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Mapping, Optional, Tuple


def _actionable_import_error(missing: str) -> ImportError:
    hint = (
        f"Optional dependency '{missing}' is required for SNN execution.\n"
        "- Try: pip install 'torch>=2.2' 'norse>=0.0.9'\n"
        "- Or (conda-forge): conda install -c conda-forge pytorch norse\n"
    )
    return ImportError(hint)


def _lazy_imports() -> Tuple[Any, Any, Any]:
    """
    Import torch and norse lazily with actionable errors.

    Returns:
        torch, lif_module, nn
    """
    try:
        torch = import_module("torch")
    except Exception as e:  # pragma: no cover - exercised only when missing
        raise _actionable_import_error("torch") from e

    try:
        lif_module = import_module("norse.torch.module.lif")
    except Exception as e:  # pragma: no cover
        raise _actionable_import_error("norse") from e

    nn = torch.nn
    return torch, lif_module, nn


def _get_cfg(path: str, cfg: Mapping[str, Any], default: Any = None) -> Any:
    """
    Fetch a nested config value using 'a.b.c' dotted path with a default.
    """
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def prepare_input(
    spikes: Any,
    *,
    device: Optional[Any] = None,
    dtype: Optional[Any] = None,
) -> Any:
    """
    Thin adapter to map encoder outputs into torch tensors with device/dtype handling.

    Args:
        spikes: torch.Tensor-like or numpy-like object. If encoder returned None (torch-free path),
                this function raises ImportError with actionable hint.
        device: torch.device or None (defaults to cpu if torch available).
        dtype: torch dtype (defaults to torch.float32).

    Returns:
        spikes_t: torch.Tensor on the requested device/dtype.

    Raises:
        ImportError: when torch is unavailable or spikes is None.
        ValueError: when spikes is not convertible to tensor or shape invalid.
    """
    if spikes is None:
        raise _actionable_import_error("torch")
    torch, _lif, _nn = _lazy_imports()
    x = spikes
    if not isinstance(x, torch.Tensor):
        try:
            x = torch.as_tensor(x)
        except Exception as e:
            raise ValueError("spikes is not convertible to a torch.Tensor.") from e
    x = x.to(device=device or torch.device("cpu"), dtype=dtype or torch.float32)
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected spikes ndim in (2,3), got {int(x.ndim)}")
    return x


def create_model(
    config: Mapping[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """
    Create a small LIF-based MLP network using Norse with configurable topology.

    Config keys (nested):
        model.unroll: bool = True
        topology.input_size: int (required unless provided at runtime by overrides)
        topology.hidden_sizes: list[int] = [128]
        topology.num_classes: int (defaults to topology.input_size if omitted)
        topology.dropout: float = 0.0
        lif.*: optional, currently passed through to metadata only for stability across Norse versions.

    Returns:
        model: torch.nn.Module
        meta:  dict with resolved topology and flags

    Raises:
        ImportError: if torch/norse are unavailable (actionable message).
        ValueError: if required topology parameters are missing.
    """
    torch, lif_module, nn = _lazy_imports()

    # Resolve topology
    input_size = _get_cfg("topology.input_size", config)
    hidden_sizes = list(_get_cfg("topology.hidden_sizes", config, [128]))
    num_classes = _get_cfg("topology.num_classes", config, input_size)
    dropout = float(_get_cfg("topology.dropout", config, 0.0))
    unroll = bool(_get_cfg("model.unroll", config, True))

    if input_size is None or int(input_size) <= 0:
        raise ValueError(
            "Missing or invalid 'topology.input_size' in model config. "
            "Provide the encoder-derived neuron count, e.g., by overriding at runtime."
        )
    if num_classes is None or int(num_classes) <= 0:
        num_classes = int(input_size)

    # Resolve Norse classes dynamically for compatibility across releases
    LIFCell = getattr(lif_module, "LIFCell", None)
    if LIFCell is None:
        # Highly unlikely with supported Norse versions
        raise ImportError("norse.torch.module.lif.LIFCell not found; please update 'norse'.")

    class LIFMLP(nn.Module):
        """
        Minimal LIF MLP with linear layers between LIF cells.

        Forward accepts:
            - x: (T, B, N) time-major spike/count/binary input
            - or x: (B, N) single step
        Returns:
            logits: (B, C)
            aux: dict { "T": int, "aggregator": str }
        """

        def __init__(
            self,
            in_features: int,
            h_sizes: list[int],
            out_features: int,
            *,
            p_dropout: float = 0.0,
            unroll_time: bool = True,
        ) -> None:
            super().__init__()
            self.unroll_time = bool(unroll_time)
            # Build interleaved Linear - LIF blocks
            dims = [int(in_features)] + [int(h) for h in h_sizes]
            self.fcs = nn.ModuleList(
                [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
            )
            self.lifs = nn.ModuleList([LIFCell() for _ in range(len(h_sizes))])
            self.dropout = nn.Dropout(p_dropout) if p_dropout and p_dropout > 0 else None
            self.head = nn.Linear(int(dims[-1]), int(out_features))
            # State placeholders (initialized as None; LIFCell handles None as zero-state)
            self._states: list[Optional[Any]] = [None for _ in range(len(self.lifs))]

        def reset_state(self, batch_size: int) -> None:
            # Let Norse LIFCell re-initialize from None on next forward
            self._states = [None for _ in range(len(self.lifs))]

        def _step(self, x_t: Any) -> Any:
            """
            Single time step through the stack. x_t: (B, N)
            """
            h = x_t
            for i, (fc, lif) in enumerate(zip(self.fcs, self.lifs)):
                h = fc(h)
                if self.dropout is not None:
                    h = self.dropout(h)
                # LIFCell returns (z, state), where z are spikes/activations
                z, s = lif(h, self._states[i])
                self._states[i] = s
                h = z
            return h  # last hidden spikes

        def forward(self, x: Any) -> Tuple[Any, Dict[str, Any]]:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            x = prepare_input(x, device=device, dtype=dtype)
            if x.ndim == 2:
                # (B, N) single-step
                self.reset_state(batch_size=int(x.shape[0]))
                last = self._step(x)
                logits = self.head(last)
                return logits, {"T": 1, "aggregator": "single"}
            # (T, B, N) time-major
            T = int(x.shape[0])
            B = int(x.shape[1])
            self.reset_state(batch_size=B)
            if not self.unroll_time:
                # Aggregate inputs first (sum over T) then one step
                agg = x.sum(dim=0)  # (B, N)
                last = self._step(agg)
                logits = self.head(last)
                return logits, {"T": T, "aggregator": "sum_then_step"}
            # Unrolled temporal forward, accumulate last hidden spikes and average
            accum = None
            for t in range(T):
                h_t = self._step(x[t])
                accum = h_t if accum is None else (accum + h_t)
            assert accum is not None
            mean_h = accum / max(T, 1)
            logits = self.head(mean_h)
            return logits, {"T": T, "aggregator": "mean_over_time"}

    model = LIFMLP(
        in_features=int(input_size),
        h_sizes=hidden_sizes,
        out_features=int(num_classes),
        p_dropout=dropout,
        unroll_time=unroll,
    )
    meta: Dict[str, Any] = {
        "topology": {
            "input_size": int(input_size),
            "hidden_sizes": [int(h) for h in hidden_sizes],
            "num_classes": int(num_classes),
            "dropout": float(dropout),
        },
        "model": {"name": _get_cfg("model.name", config, "norse_lif"), "unroll": bool(unroll)},
        "backend": "norse",
    }
    return model, meta


__all__ = ["create_model", "prepare_input"]