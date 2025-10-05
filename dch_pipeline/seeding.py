# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Determinism and random seed control utilities.

Provides:
- set_global_seeds: set Python, NumPy, and PyTorch seeds consistently
- enable_torch_determinism: toggle deterministic algorithms and cuDNN flags
- environment_seed_context: context manager to temporarily set seeds and restore after

References
- Usage in pipeline and runners; see README and docs/REPRODUCIBILITY.md
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Mapping, Optional

import numpy as np
# Optional torch import for environments without PyTorch
try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


def set_global_seeds(
    seed: int,
    *,
    numpy_seed: Optional[int] = None,
    torch_seed: Optional[int] = None,
    extra: Optional[Mapping[str, int]] = None,
) -> None:
    """
    Set seeds for Python's random, NumPy, and PyTorch (CPU and all CUDA devices).

    Args:
        seed: base seed for Python random
        numpy_seed: if provided, overrides base seed for NumPy
        torch_seed: if provided, overrides base seed for PyTorch
        extra: optional mapping for additional libraries (namespaced by caller)
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np_seed = numpy_seed if numpy_seed is not None else seed
    np.random.seed(np_seed)

    th_seed = torch_seed if torch_seed is not None else seed
    if TORCH_AVAILABLE:
        torch.manual_seed(th_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(th_seed)


def enable_torch_determinism(
    *,
    deterministic: bool = True,
    cudnn_deterministic: Optional[bool] = None,
    cudnn_benchmark: Optional[bool] = None,
    warn_only: bool = False,
) -> None:
    """
    Configure PyTorch's deterministic algorithms and cuDNN behavior.

    Args:
        deterministic: torch.use_deterministic_algorithms(deterministic)
        cudnn_deterministic: sets torch.backends.cudnn.deterministic if not None
        cudnn_benchmark: sets torch.backends.cudnn.benchmark if not None
        warn_only: when True, PyTorch will warn instead of error for nondeterministic ops
    """
    if not TORCH_AVAILABLE:
        return

    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=bool(warn_only))
    except Exception:
        # Older PyTorch versions may not support warn_only
        torch.use_deterministic_algorithms(bool(deterministic))  # type: ignore[arg-type]

    if cudnn_deterministic is not None:
        torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    if cudnn_benchmark is not None:
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


@contextmanager
def environment_seed_context(
    seed: int,
    *,
    deterministic: bool = True,
    numpy_seed: Optional[int] = None,
    torch_seed: Optional[int] = None,
):
    """
    Context manager to set seeds/determinism, then restore cuDNN flags after.

    Note:
        Python/random and NumPy do not provide an officially supported way to snapshot/restore
        global RNG state portably; if exact restoration is needed, callers should snapshot
        random.getstate() and np.random.get_state() themselves.
    """
    # Snapshot cuDNN flags to restore later (only if torch is available)
    if TORCH_AVAILABLE:
        cudnn_det_old = torch.backends.cudnn.deterministic
        cudnn_bm_old = torch.backends.cudnn.benchmark
    else:
        cudnn_det_old = None
        cudnn_bm_old = None

    set_global_seeds(seed, numpy_seed=numpy_seed, torch_seed=torch_seed)
    # Default deterministic configuration for fairness across runs
    enable_torch_determinism(
        deterministic=deterministic,
        cudnn_deterministic=True if deterministic else False,
        cudnn_benchmark=False if deterministic else True,
    )
    try:
        yield
    finally:
        # Restore cuDNN flags (only if torch is available)
        if TORCH_AVAILABLE and cudnn_det_old is not None and cudnn_bm_old is not None:
            torch.backends.cudnn.deterministic = cudnn_det_old
            torch.backends.cudnn.benchmark = cudnn_bm_old


__all__ = ["set_global_seeds", "enable_torch_determinism", "environment_seed_context"]