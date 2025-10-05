# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Event-to-spike encoders.

Implements:
- SimpleBinnerEncoder: bins events into fixed-size temporal windows to produce a
  time-major spike tensor of shape (T, B=1, N), where N is the number of unique
  neuron ids observed in the window.

Notes
- This minimal encoder is backend-neutral and follows the Encoder Protocol from
  dch_snn.interface. It is suitable for CPU evaluation and unit tests.
- For camera event datasets (DVS), a spatial encoder variant can be added later
  to produce tensors of shape (T, B, H, W, C), using EncoderConfig.shape.

References
- Interface: dch_snn.interface (Encoder, EncoderConfig)
- Types: dch_core.interfaces (Event, Window, Timestamp, NeuronId)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create a minimal torch-like interface for fallback
    class torch:
        @staticmethod
        def zeros(*args, **kwargs):
            return None
        
        class device:
            def __init__(self, name):
                self.name = name

from dch_core.interfaces import Event, Window, Timestamp, NeuronId
from dch_snn.interface import Encoder, EncoderConfig


def _compute_T(window: Window, time_bin: int) -> int:
    """Compute number of time bins T covering [t0, t1] inclusive with bin width time_bin."""
    t0, t1 = window
    if t1 < t0:
        return 0
    span = (t1 - t0)
    return int(span // time_bin) + 1


def _bin_index(t: Timestamp, t0: Timestamp, time_bin: int) -> int:
    """Map a timestamp t to its bin index relative to t0."""
    return int((t - t0) // time_bin)


@dataclass
class SimpleBinnerEncoder(Encoder):
    """
    A simple event-to-spike binner that:
    - infers the set of active neuron ids within the window
    - assigns a column per neuron id (sorted for determinism)
    - sets spikes[t_bin, 0, n_idx] = 1 when an event occurs

    Configuration
    - EncoderConfig.time_bin: bin width (default 1000)
    - EncoderConfig.normalize: if True, converts counts per bin to {0,1} (binary spikes).
      Set to False to produce integer counts per bin.

    Output
    - spikes: torch.Tensor of shape (T, 1, N)
    - meta:
        {
          "neuron_index": { neuron_id (int) -> column index (int) },
          "T": int,
          "N": int,
          "t0": int,
          "t1": int,
          "time_bin": int,
        }
    """

    config: EncoderConfig

    def reset(self) -> None:
        # Stateless encoder; nothing to reset.
        return

    def encode(
        self,
        events: Sequence[Event],
        window: Window,
        device: Any,  # torch.device when available
    ) -> Tuple[Any, Mapping[str, Any]]:  # torch.Tensor when available
        if not TORCH_AVAILABLE:
            # Fallback when torch not available - return dummy data
            t0, t1 = window
            return None, {
                "neuron_index": {},
                "T": 0,
                "N": 0,
                "t0": int(t0),
                "t1": int(t1),
                "time_bin": int(self.config.time_bin),
                "torch_available": False
            }
            
        t0, t1 = window
        T = _compute_T(window, self.config.time_bin)
        if T <= 0:
            # Return an empty tensor (0, 1, 0)
            empty = torch.zeros((0, 1, 0), dtype=torch.float32, device=device)
            return empty, {"neuron_index": {}, "T": 0, "N": 0, "t0": int(t0), "t1": int(t1), "time_bin": int(self.config.time_bin)}

        # Filter events to the window and collect active neuron ids
        active_ids: set[NeuronId] = set()
        filtered: list[Event] = []
        for e in events:
            if t0 <= e.t <= t1:
                filtered.append(e)
                active_ids.add(e.neuron_id)

        # Deterministic neuron id ordering for column assignment
        neuron_list = sorted(active_ids)
        N = len(neuron_list)
        neuron_index: Dict[int, int] = {int(nid): idx for idx, nid in enumerate(neuron_list)}

        # Allocate spikes: time-major (T, B=1, N)
        spikes = torch.zeros((T, 1, N), dtype=torch.float32, device=device) if N > 0 else torch.zeros((T, 1, 0), dtype=torch.float32, device=device)

        # Populate spikes
        for e in filtered:
            col = neuron_index.get(int(e.neuron_id))
            if col is None:
                # Should not happen given active_ids collection; safeguard for robustness
                continue
            bin_idx = _bin_index(e.t, t0, self.config.time_bin)
            if 0 <= bin_idx < T:
                spikes[bin_idx, 0, col] += 1.0

        # Normalize to binary spikes if requested
        if self.config.normalize and N > 0:
            spikes.clamp_(0.0, 1.0)

        meta: Dict[str, Any] = {
            "neuron_index": neuron_index,
            "T": int(T),
            "N": int(N),
            "t0": int(t0),
            "t1": int(t1),
            "time_bin": int(self.config.time_bin),
            "torch_available": True
        }
        return spikes, meta


__all__ = ["SimpleBinnerEncoder"]