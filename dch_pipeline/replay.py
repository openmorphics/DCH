# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Deterministic replay harness utilities (stdlib-only).

This module provides:
- set_global_seeds(seeds: dch_core.interfaces.SeedConfig) -> dict
  Set seeds for Python's random, NumPy (if available), and torch (if available).
  Returns a dict of actually applied seeds:
      {"python": int, "numpy": int | None, "torch": int | None}
  Notes:
  - Uses only the Python standard library and optional imports resolved at runtime.
  - If NumPy or torch are not available, their entries are set to None and silently skipped.
  - No environment variables are modified (e.g., PYTHONHASHSEED is not set here).
  - No I/O side-effects.

- get_environment_fingerprint() -> dch_core.interfaces.EnvironmentFingerprint
  Capture a lightweight environment fingerprint including:
    * python_version (sys.version)
    * platform (platform.platform())
    * If torch is available:
        - torch_version
        - cuda (torch.version.cuda if present)
        - cudnn (torch.backends.cudnn.version() if available)
    * packages: optional mapping with discovered package versions (currently numpy/torch if found)

Determinism guarantees:
- For Python's stdlib RNG: random.seed(seeds.python) produces deterministic sequences.
- For NumPy (if present): np.random.seed(seeds.numpy) produces deterministic sequences for the legacy global RNG.
- For torch (if present): torch.manual_seed(seeds.torch) and torch.cuda.manual_seed_all(seeds.torch) (if CUDA available).
  This does not toggle any deterministic algorithm flags. Users needing stricter determinism can use dch_pipeline.seeding
  utilities that manage CuDNN and algorithm flags explicitly.

These utilities are designed to be side-effect minimal and import-safe across environments where optional
libraries may be absent.
"""

from __future__ import annotations

import importlib
import importlib.util as _ilu
import platform as _platform
import random
import sys
from typing import Dict, Optional

from dch_core.interfaces import EnvironmentFingerprint, SeedConfig


def set_global_seeds(seeds: SeedConfig) -> Dict[str, Optional[int]]:
    """
    Set Python random, NumPy (if available), and torch (if available) seeds deterministically.

    Args:
        seeds: SeedConfig with fields:
            - python: int seed for Python's random
            - numpy: int seed for NumPy (if installed)
            - torch: int seed for torch (if installed)
            - extra: Mapping[str, int] (ignored here; reserved for caller-specific namespaces)

    Returns:
        Dict with actually applied seeds:
            {"python": int, "numpy": int | None, "torch": int | None}

    Behavior:
    - Uses importlib.util.find_spec to detect optional libraries.
    - Silently skips seeding when an optional library is not available.
    - Avoids any file I/O or environment variable mutations.
    """
    applied: Dict[str, Optional[int]] = {"python": int(seeds.python), "numpy": None, "torch": None}

    # Python stdlib RNG
    try:
        random.seed(int(seeds.python))
    except Exception:
        # Extremely defensive; if this fails, leave state as-is
        pass

    # NumPy (optional)
    try:
        if _ilu.find_spec("numpy") is not None:  # type: ignore
            np = importlib.import_module("numpy")  # type: ignore[assignment]
            try:
                np.random.seed(int(seeds.numpy))
                applied["numpy"] = int(seeds.numpy)
            except Exception:
                applied["numpy"] = None
    except Exception:
        applied["numpy"] = None

    # torch (optional)
    try:
        if _ilu.find_spec("torch") is not None:  # type: ignore
            torch = importlib.import_module("torch")  # type: ignore
            try:
                torch.manual_seed(int(seeds.torch))
                # CUDA seeding if available
                if hasattr(torch, "cuda") and callable(getattr(torch.cuda, "is_available", None)) and torch.cuda.is_available():  # type: ignore[attr-defined]
                    try:
                        torch.cuda.manual_seed_all(int(seeds.torch))  # type: ignore[attr-defined]
                    except Exception:
                        # CUDA seeding best-effort
                        pass
                applied["torch"] = int(seeds.torch)
            except Exception:
                applied["torch"] = None
    except Exception:
        applied["torch"] = None

    return applied


def get_environment_fingerprint() -> EnvironmentFingerprint:
    """
    Capture a lightweight, stdlib-only environment fingerprint.

    Fields populated:
    - python_version: sys.version (single-line normalized)
    - platform: platform.platform()
    - torch_version/cuda/cudnn: only if torch is importable
    - packages: includes discovered versions for {"numpy": __version__, "torch": __version__} when present

    Returns:
        EnvironmentFingerprint dataclass instance.
    """
    py_ver = " ".join(str(sys.version).split())  # normalize whitespace/newlines
    plat = _platform.platform()
    torch_version: Optional[str] = None
    cuda: Optional[str] = None
    cudnn: Optional[str] = None
    packages: Dict[str, str] = {}

    # NumPy (optional)
    try:
        if _ilu.find_spec("numpy") is not None:  # type: ignore
            np = importlib.import_module("numpy")  # type: ignore[assignment]
            ver = getattr(np, "__version__", None)
            if isinstance(ver, str):
                packages["numpy"] = ver
    except Exception:
        pass

    # torch (optional)
    try:
        if _ilu.find_spec("torch") is not None:  # type: ignore
            torch = importlib.import_module("torch")  # type: ignore
            tver = getattr(torch, "__version__", None)
            if isinstance(tver, str):
                torch_version = tver
                packages["torch"] = tver
            # CUDA version string
            try:
                cuda_ver = getattr(getattr(torch, "version", None), "cuda", None)
                if isinstance(cuda_ver, str):
                    cuda = cuda_ver
            except Exception:
                cuda = None
            # cuDNN version (if backend present)
            try:
                backends = getattr(torch, "backends", None)
                if backends is not None:
                    cudnn_mod = getattr(backends, "cudnn", None)
                    if cudnn_mod is not None and hasattr(cudnn_mod, "version"):
                        v = cudnn_mod.version()  # type: ignore[attr-defined]
                        if isinstance(v, int):
                            cudnn = str(v)
                        elif isinstance(v, str):
                            cudnn = v
            except Exception:
                cudnn = None
    except Exception:
        torch_version = None
        cuda = None
        cudnn = None

    return EnvironmentFingerprint(
        python_version=py_ver,
        platform=plat,
        torch_version=torch_version,
        cuda=cuda,
        cudnn=cudnn,
        packages=packages,
    )


__all__ = ["set_global_seeds", "get_environment_fingerprint"]