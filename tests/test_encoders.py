# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Encoders tests (torch-free by default).

Conventions:
- Use importlib.util.find_spec to gate optional deps.
- Skip clearly if numpy is missing.
- Do NOT require torch; when torch is unavailable, validate the torch-free fallback metadata.
- Keep runtime small and deterministic.
"""

from __future__ import annotations

import importlib.util as ilu
import random
from typing import Any, Sequence

import pytest

# Gate on numpy presence per global conventions
if ilu.find_spec("numpy") is None:  # pragma: no cover
    pytest.skip(
        "Skipping encoder tests: numpy not installed. "
        "Install with 'pip install numpy' or 'conda install -c conda-forge numpy'.",
        allow_module_level=True,
    )
import numpy as np  # type: ignore

from dch_core.interfaces import Event, Window
from dch_snn.interface import EncoderConfig
from dch_data.encoders import SimpleBinnerEncoder


def _has_module(name: str) -> bool:
    return ilu.find_spec(name) is not None


def _maybe_torch_device() -> Any:
    if not _has_module("torch"):
        return None
    import torch  # type: ignore

    return torch.device("cpu")


def _expected_T(window: Window, time_bin: int) -> int:
    t0, t1 = window
    if t1 < t0:
        return 0
    return (t1 - t0) // time_bin + 1


def _seed():
    random.seed(0)
    np.random.seed(0)


def _build_events() -> Sequence[Event]:
    # Monotonic timestamps; two neurons {0,1} inside window, one event outside window
    return [
        Event(neuron_id=0, t=100),
        Event(neuron_id=1, t=1100),
        Event(neuron_id=0, t=2500),
        Event(neuron_id=1, t=3500),  # outside [0, 3000]
    ]


def test_simple_binner_deterministic_and_meta_shape():
    """
    Validate:
    - meta fields and determinism with same input/params
    - spikes shape and determinism when torch is available
    """
    _seed()

    time_bin = 1000
    window: Window = (0, 3000)
    enc = SimpleBinnerEncoder(EncoderConfig(time_bin=time_bin, normalize=True))
    events = _build_events()

    device = _maybe_torch_device()
    spikes1, meta1 = enc.encode(events=events, window=window, device=device)
    spikes2, meta2 = enc.encode(events=events, window=window, device=device)

    # Determinism of metadata across runs
    assert meta1["time_bin"] == time_bin and meta2["time_bin"] == time_bin
    assert meta1["t0"] == window[0] and meta1["t1"] == window[1]
    assert meta2["t0"] == window[0] and meta2["t1"] == window[1]

    if _has_module("torch"):
        import torch  # type: ignore

        expected_T = _expected_T(window, time_bin)
        expected_N = 2  # neuron ids {0,1} within window

        # Deterministic neuron index mapping
        assert meta1["neuron_index"] == meta2["neuron_index"] == {0: 0, 1: 1}

        assert isinstance(spikes1, torch.Tensor) and isinstance(spikes2, torch.Tensor)
        assert tuple(spikes1.shape) == (expected_T, 1, expected_N)
        assert tuple(spikes2.shape) == (expected_T, 1, expected_N)
        # Exact equality: deterministic fill + clamp to {0,1}
        assert torch.sum(torch.abs(spikes1 - spikes2)).item() == 0.0
    else:
        # Torch-free fallback path: spikes is None and torch_available=False
        assert spikes1 is None and spikes2 is None
        assert meta1.get("torch_available") is False
        assert meta2.get("torch_available") is False
        # Metadata determinism
        assert meta1 == meta2


def test_simple_binner_windowing_bins():
    """
    Validate that events land in expected bins when torch is available.
    If torch is unavailable, skip with a clear reason (window slicing verification needs tensor).
    """
    if not _has_module("torch"):  # pragma: no cover
        pytest.skip("torch not installed; windowing verification requires tensor output")

    import torch  # type: ignore

    _seed()

    time_bin = 1000
    window: Window = (0, 3000)
    enc = SimpleBinnerEncoder(EncoderConfig(time_bin=time_bin, normalize=True))
    events = _build_events()
    device = torch.device("cpu")

    spikes, meta = enc.encode(events=events, window=window, device=device)
    assert isinstance(spikes, torch.Tensor)
    idx = meta["neuron_index"]

    # Only in-window events considered: t=100 (n0), 1100 (n1), 2500 (n0)
    t0 = window[0]
    in_window = [(0, 100), (1, 1100), (0, 2500)]
    for nid, t in in_window:
        bin_idx = (t - t0) // time_bin
        col = idx[nid]
        val = float(spikes[bin_idx, 0, col].item())
        assert val > 0.0, f"Expected nonzero spike at bin {bin_idx}, neuron {nid}"


def test_simple_binner_fallback_meta_when_torch_absent():
    """
    Torch-free default path:
    - encode returns (None, meta) with torch_available=False
    - meta contains coherent window/time_bin fields
    """
    if _has_module("torch"):  # pragma: no cover
        pytest.skip("torch installed; this test validates torch-free fallback only")

    _seed()

    time_bin = 1000
    window: Window = (0, 3000)
    enc = SimpleBinnerEncoder(EncoderConfig(time_bin=time_bin, normalize=True))
    events = _build_events()

    spikes, meta = enc.encode(events=events, window=window, device=None)
    assert spikes is None
    assert meta.get("torch_available") is False
    # Meta is coherent even if shapes are not computed in torch-free path
    assert meta["t0"] == window[0]
    assert meta["t1"] == window[1]
    assert meta["time_bin"] == time_bin