# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Event-to-spike encoders.

Implements:
- SimpleBinnerEncoder: bins events into fixed-size temporal windows to produce a
  time-major spike tensor of shape (T, B=1, N), where N is the number of unique
  neuron ids observed in the window.
- LatencyEncoder: encodes earliest event per neuron into a single spike at the
  corresponding time bin (latency code).
- RateEncoder: encodes binned firing rates per neuron over time, with optional
  Poisson sampling when torch is available.

Notes
- Encoders follow the Encoder Protocol from dch_snn.interface and are suitable for
  CPU evaluation and unit tests.
- Import-safety: No optional dependencies (torch/numpy) are imported at module
  top-level. Availability is detected lazily with importlib.util.find_spec.
- Torch-absent behavior: encode(...) returns (None, meta) with torch_available=False
  and coherent window metadata.

References
- Interface: dch_snn.interface (Encoder, EncoderConfig)
- Types: dch_core.interfaces (Event, Window, Timestamp, NeuronId)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import importlib.util as ilu

from dch_core.interfaces import Event, Window, Timestamp, NeuronId
from dch_snn.interface import Encoder, EncoderConfig


# -------------------------
# Optional dependency gates
# -------------------------


def _has_module(name: str) -> bool:
    """Return True if a module can be imported, without importing it."""
    return ilu.find_spec(name) is not None


def _get_torch() -> Optional[Any]:
    """
    Lazily import torch if available, else return None.
    Import via importlib to avoid static import-time errors when torch is absent.
    """
    if not _has_module("torch"):
        return None
    import importlib
    try:
        return importlib.import_module("torch")
    except Exception:
        return None


# -------------------------
# Shared helpers (private)
# -------------------------


def _compute_T(window: Window, bin_size_us: int) -> int:
    """
    Compute number of time bins T covering [t0, t1] inclusive with bin width bin_size_us.
    """
    t0, t1 = window
    if t1 < t0:
        return 0
    span = (t1 - t0)
    return int(span // bin_size_us) + 1


def _timestamp_to_bin(ts: Timestamp, t0: Timestamp, bin_size_us: int) -> int:
    """Map a timestamp to its bin index relative to t0."""
    return int((ts - t0) // bin_size_us)


def _compute_window(
    events: Sequence[Event],
    t_start: Optional[int],
    t_end: Optional[int],
) -> Window:
    """
    Decide a coherent closed window [t0, t1].

    Rules:
    - If both t_start and t_end are provided:
        - ensure t0 <= t1 (swap if necessary)
    - If only t_start is provided:
        - t0 = t_start; t1 = max(events.t) if events else t_start
    - If only t_end is provided:
        - t0 = min(events.t) if events else t_end; t1 = t_end
    - If neither is provided:
        - If events exist: [min_t, max_t]
        - Else: [0, 0] (degenerate)
    """
    if t_start is not None and t_end is not None:
        t0 = int(t_start)
        t1 = int(t_end)
        if t1 < t0:
            t0, t1 = t1, t0
        return (t0, t1)

    if t_start is not None:
        t0 = int(t_start)
        if events:
            t1 = max(int(e.t) for e in events)
            if t1 < t0:
                t1 = t0
        else:
            t1 = t0
        return (t0, t1)

    if t_end is not None:
        t1 = int(t_end)
        if events:
            t0 = min(int(e.t) for e in events)
            if t1 < t0:
                t0 = t1
        else:
            t0 = t1
        return (t0, t1)

    if events:
        min_t = min(int(e.t) for e in events)
        max_t = max(int(e.t) for e in events)
        return (min_t, max_t)

    return (0, 0)


def _bin_edges(t0: Timestamp, t1: Timestamp, bin_size_us: int) -> Sequence[Timestamp]:
    """
    Return the left edges of bins covering [t0, t1], inclusive.

    Example:
        t0=0, t1=3000, bin=1000 -> [0, 1000, 2000, 3000]
    """
    T = _compute_T((t0, t1), bin_size_us)
    return [t0 + i * bin_size_us for i in range(T)]


def _resolve_num_neurons(
    events: Sequence[Event], num_neurons_override: Optional[int] = None
) -> Tuple[List[int], Dict[int, int]]:
    """
    Resolve neuron id set and mapping.
    - Gather observed neuron ids from events
    - If override is given, include ids [0..override-1] as baseline
    - Return (sorted_neuron_ids, neuron_index_map)
    """
    observed: Set[int] = set(int(e.neuron_id) for e in events)
    baseline: Set[int] = set(range(int(num_neurons_override))) if num_neurons_override is not None else set()
    all_ids = observed | baseline
    neuron_list = sorted(all_ids)
    neuron_index = {nid: idx for idx, nid in enumerate(neuron_list)}
    return neuron_list, neuron_index


# -------------------------
# Encoders
# -------------------------


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
        torch = _get_torch()
        t0, t1 = window

        if torch is None:
            # Fallback when torch not available - return metadata only
            return None, {
                "neuron_index": {},
                "T": 0,
                "N": 0,
                "t0": int(t0),
                "t1": int(t1),
                "time_bin": int(self.config.time_bin),
                "torch_available": False,
            }

        T = _compute_T(window, int(self.config.time_bin))
        if T <= 0:
            # Return an empty tensor (0, 1, 0)
            kwargs = {"device": device} if device is not None else {}
            empty = torch.zeros((0, 1, 0), dtype=torch.float32, **kwargs)
            return empty, {
                "neuron_index": {},
                "T": 0,
                "N": 0,
                "t0": int(t0),
                "t1": int(t1),
                "time_bin": int(self.config.time_bin),
                "torch_available": True,
            }

        # Filter events to the window and collect active neuron ids
        active_ids: Set[NeuronId] = set()
        filtered: List[Event] = []
        for e in events:
            if t0 <= e.t <= t1:
                filtered.append(e)
                active_ids.add(e.neuron_id)

        # Deterministic neuron id ordering for column assignment
        neuron_list = sorted(int(nid) for nid in active_ids)
        N = len(neuron_list)
        neuron_index: Dict[int, int] = {nid: idx for idx, nid in enumerate(neuron_list)}

        # Allocate spikes: time-major (T, B=1, N)
        kwargs = {"device": device} if device is not None else {}
        spikes = (
            torch.zeros((T, 1, N), dtype=torch.float32, **kwargs)
            if N > 0
            else torch.zeros((T, 1, 0), dtype=torch.float32, **kwargs)
        )

        # Populate spikes
        for e in filtered:
            col = neuron_index.get(int(e.neuron_id))
            if col is None:
                continue
            bin_idx = _timestamp_to_bin(e.t, t0, int(self.config.time_bin))
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
            "torch_available": True,
        }
        return spikes, meta


@dataclass
class LatencyEncoder(Encoder):
    """
    Latency encoder: Encode earliest event per neuron as a single spike at its first
    time bin.

    Parameters
    - bin_size_us: int. Temporal bin width.
    - t_start: Optional[int]. If provided, lower bound of window (inclusive).
    - t_end: Optional[int]. If provided, upper bound of window (inclusive).
    - clamp_out_of_window: bool (default True). If a neuron's earliest event lies
      outside [t0,t1], clamp to nearest boundary bin when True, otherwise skip.
    - num_neurons: Optional[int]. Override baseline neuron ids 0..num_neurons-1.

    Output
    - Torch-present: torch.int8 tensor of shape (T, 1, N) with one spike at the earliest
      bin for each neuron that has at least one event; zeros otherwise.
    - Torch-absent: (None, meta) with torch_available=False and coherent metadata.
    """

    bin_size_us: int = 1000
    t_start: Optional[int] = None
    t_end: Optional[int] = None
    clamp_out_of_window: bool = True
    num_neurons: Optional[int] = None

    def reset(self) -> None:
        return

    def encode(
        self,
        events: Sequence[Event],
        window: Window,
        device: Any,
    ) -> Tuple[Any, Mapping[str, Any]]:
        # Choose window: honor explicit t_start/t_end when provided, else use input window
        if self.t_start is not None or self.t_end is not None:
            t0, t1 = _compute_window(events, self.t_start, self.t_end)
        else:
            t0, t1 = window

        T = _compute_T((t0, t1), int(self.bin_size_us))

        # Resolve neuron ids deterministically
        neuron_list, neuron_index = _resolve_num_neurons(events, self.num_neurons)
        N = len(neuron_list)

        torch = _get_torch()
        meta: Dict[str, Any] = {
            "encoder_name": "latency",
            "neuron_index": neuron_index,
            "T": int(T),
            "N": int(N),
            "t0": int(t0),
            "t1": int(t1),
            "bin_size_us": int(self.bin_size_us),
            "num_bins": int(T),
        }

        if torch is None:
            meta["torch_available"] = False
            return None, meta

        kwargs = {"device": device} if device is not None else {}
        spikes = torch.zeros((T, 1, N), dtype=torch.int8, **kwargs) if N > 0 else torch.zeros((T, 1, 0), dtype=torch.int8, **kwargs)

        if T <= 0 or N == 0:
            meta["torch_available"] = True
            return spikes, meta

        # For each neuron, place a single spike at earliest event bin with clamping logic
        # Build per-neuron earliest timestamp
        earliest: Dict[int, Optional[int]] = {nid: None for nid in neuron_list}
        for e in events:
            nid = int(e.neuron_id)
            if nid not in neuron_index:
                continue
            ts = int(e.t)
            prev = earliest[nid]
            if prev is None or ts < prev:
                earliest[nid] = ts

        for nid in neuron_list:
            ts = earliest.get(nid)
            if ts is None:
                continue
            # Clamp or skip based on window
            if ts < t0:
                if not self.clamp_out_of_window:
                    continue
                ts_eff = t0
            elif ts > t1:
                if not self.clamp_out_of_window:
                    continue
                ts_eff = t1
            else:
                ts_eff = ts

            bin_idx = _timestamp_to_bin(ts_eff, t0, int(self.bin_size_us))
            if 0 <= bin_idx < T:
                col = neuron_index[nid]
                spikes[bin_idx, 0, col] = 1  # single spike

        meta["torch_available"] = True
        return spikes, meta


@dataclass
class RateEncoder(Encoder):
    """
    Rate encoder: Encode binned firing rate per neuron over time bins.
    Optionally Poisson-sample spikes using the bin rate as lambda when torch is available.

    Parameters
    - bin_size_us: int. Temporal bin width.
    - t_start: Optional[int]. Lower bound of window (inclusive).
    - t_end: Optional[int]. Upper bound of window (inclusive).
    - normalize: str in {"none", "per_neuron"}; default "per_neuron".
        * "none": return raw counts per bin as float
        * "per_neuron": divide each neuron's per-bin counts by its max across T (if max>0)
    - poisson: bool (default False). If True and torch available, sample Poisson(lam=rates)
      per bin. Deterministic only when seed is set.
    - seed: Optional[int]. If provided and poisson=True, torch.manual_seed(seed) is set within encode.
    - num_neurons: Optional[int]. Override baseline neuron ids 0..num_neurons-1.

    Output
    - Torch-present:
        * poisson=True: int tensor (T, 1, N) with sampled counts
        * poisson=False: float tensor (T, 1, N) with rates
    - Torch-absent: (None, meta) with torch_available=False and rate summary
    """

    bin_size_us: int = 1000
    t_start: Optional[int] = None
    t_end: Optional[int] = None
    normalize: str = "per_neuron"
    poisson: bool = False
    seed: Optional[int] = None
    num_neurons: Optional[int] = None

    def reset(self) -> None:
        return

    def encode(
        self,
        events: Sequence[Event],
        window: Window,
        device: Any,
    ) -> Tuple[Any, Mapping[str, Any]]:
        # Choose window
        if self.t_start is not None or self.t_end is not None:
            t0, t1 = _compute_window(events, self.t_start, self.t_end)
        else:
            t0, t1 = window

        T = _compute_T((t0, t1), int(self.bin_size_us))
        neuron_list, neuron_index = _resolve_num_neurons(events, self.num_neurons)
        N = len(neuron_list)

        # Build counts[T][N] with pure Python
        counts: List[List[int]] = [[0 for _ in range(N)] for _ in range(max(T, 0))]
        in_window_events = 0
        for e in events:
            ts = int(e.t)
            if ts < t0 or ts > t1 or N == 0 or T <= 0:
                continue
            col = neuron_index.get(int(e.neuron_id))
            if col is None:
                # Robust: include events with ids beyond override by extending mapping? We fixed mapping earlier;
                # if an unseen id appears (shouldn't given mapping), skip safely.
                continue
            bin_idx = _timestamp_to_bin(ts, t0, int(self.bin_size_us))
            if 0 <= bin_idx < T:
                counts[bin_idx][col] += 1
                in_window_events += 1

        # Normalize as requested
        rates: List[List[float]]
        if self.normalize == "none":
            rates = [[float(v) for v in row] for row in counts]
        else:
            # per_neuron normalization
            per_neuron_max = [0 for _ in range(N)]
            for j in range(N):
                per_neuron_max[j] = max((counts[i][j] for i in range(T)), default=0) if T > 0 else 0
            rates = []
            for i in range(T):
                row: List[float] = []
                for j in range(N):
                    m = per_neuron_max[j]
                    row.append((counts[i][j] / m) if m > 0 else 0.0)
                rates.append(row)

        torch = _get_torch()
        meta: Dict[str, Any] = {
            "encoder_name": "rate",
            "neuron_index": neuron_index,
            "T": int(T),
            "N": int(N),
            "t0": int(t0),
            "t1": int(t1),
            "bin_size_us": int(self.bin_size_us),
            "num_bins": int(T),
            "normalize": str(self.normalize),
            "poisson": bool(self.poisson),
        }

        if torch is None:
            meta["torch_available"] = False
            # Provide a small summary to aid downstream logic/tests in torch-free mode
            meta["rate_summary"] = {
                "total_events_in_window": int(in_window_events),
                "per_neuron_totals": {nid: sum(counts[i][idx] for i in range(T)) if (T > 0 and N > 0) else 0
                                      for nid, idx in neuron_index.items()},
            }
            return None, meta

        kwargs = {"device": device} if device is not None else {}
        if T <= 0 or N == 0:
            # Degenerate empty shapes
            if self.poisson:
                empty = torch.zeros((0, 1, 0), dtype=torch.int32, **kwargs)
            else:
                empty = torch.zeros((0, 1, 0), dtype=torch.float32, **kwargs)
            meta["torch_available"] = True
            return empty, meta

        # Materialize rates tensor (T,N)
        rates_tensor = torch.tensor(rates, dtype=torch.float32, **kwargs)

        if self.poisson:
            if self.seed is not None:
                torch.manual_seed(int(self.seed))
            # torch.poisson returns float tensor; cast to int
            sampled = torch.poisson(rates_tensor).to(dtype=torch.int32)
            spikes = sampled.unsqueeze(1)  # (T,1,N)
            meta["torch_available"] = True
            return spikes, meta

        # Deterministic float rates path
        spikes = rates_tensor.unsqueeze(1)  # (T,1,N)
        meta["torch_available"] = True
        return spikes, meta


# -------------------------
# Factory
# -------------------------


def get_encoder(name: str, **kwargs: Any) -> Encoder:
    """
    Factory for encoders.

    Args:
        name:
            - "binner": SimpleBinnerEncoder
            - "latency": LatencyEncoder
            - "rate": RateEncoder
        **kwargs:
            - For "binner": time_bin (int), normalize (bool)
            - For "latency": bin_size_us, t_start, t_end, clamp_out_of_window, num_neurons
            - For "rate": bin_size_us, t_start, t_end, normalize, poisson, seed, num_neurons

    Returns:
        Encoder instance.
    """
    key = name.strip().lower()
    if key in ("binner", "simple", "simple_binner", "simple-binner"):
        time_bin = int(kwargs.get("time_bin", kwargs.get("bin_size_us", 1000)))
        normalize = bool(kwargs.get("normalize", True))
        cfg = EncoderConfig(time_bin=time_bin, normalize=normalize)
        return SimpleBinnerEncoder(cfg)

    if key == "latency":
        return LatencyEncoder(
            bin_size_us=int(kwargs.get("bin_size_us", 1000)),
            t_start=kwargs.get("t_start"),
            t_end=kwargs.get("t_end"),
            clamp_out_of_window=bool(kwargs.get("clamp_out_of_window", True)),
            num_neurons=kwargs.get("num_neurons"),
        )

    if key == "rate":
        normalize = str(kwargs.get("normalize", "per_neuron"))
        if normalize not in ("none", "per_neuron"):
            raise ValueError("RateEncoder.normalize must be 'none' or 'per_neuron'")
        return RateEncoder(
            bin_size_us=int(kwargs.get("bin_size_us", 1000)),
            t_start=kwargs.get("t_start"),
            t_end=kwargs.get("t_end"),
            normalize=normalize,
            poisson=bool(kwargs.get("poisson", False)),
            seed=kwargs.get("seed"),
            num_neurons=kwargs.get("num_neurons"),
        )

    raise ValueError(f"Unknown encoder name: {name!r}")


__all__ = ["SimpleBinnerEncoder", "LatencyEncoder", "RateEncoder", "get_encoder"]