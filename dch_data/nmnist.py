# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
N-MNIST dataset loader with optional dependency gating (tonic, numpy).

Goals
- Import-safe without tonic/numpy installed (no top-level optional imports).
- Lazy-import optional deps at call-time with friendly, actionable errors.
- Minimal, torch-free API focused on gating behavior.

Public API
- NmnistLoader: minimal loader with lazy imports and load_one(index) method.

Usage
    loader = NmnistLoader(root="~/.cache/tonic", split="train", download=False)
    sample = loader.load_one(0)  # Raises ImportError with install hints if deps missing

Notes
- This module intentionally avoids importing 'tonic' or 'numpy' at module top-level.
- When attempting to use load_one(), the required optional dependencies are imported
  via importlib.import_module with clear error messages if unavailable.
"""

from __future__ import annotations

from typing import Any
import importlib
import os


def _lazy_import(module_name: str, context: str) -> Any:
    """
    Import an optional dependency at runtime with actionable error messages.

    Args:
        module_name: Name of the module to import (e.g., "tonic", "numpy").
        context: Human-friendly context for the error message (e.g., "N-MNIST loader").

    Returns:
        Imported module object.

    Raises:
        ImportError: If the module cannot be imported. The message includes pip and conda hints.
    """
    try:
        return importlib.import_module(module_name)
    except Exception as e:  # pragma: no cover - exercise via tests
        raise ImportError(
            f"Optional dependency '{module_name}' is required for {context}. Install via:\n"
            f"- pip: pip install {module_name}\n"
            f"- conda: conda install -c conda-forge {module_name}"
        ) from e


class NmnistLoader:
    """
    Minimal N-MNIST loader that constructs the tonic dataset on demand and loads a single item.

    This class performs lazy imports of optional dependencies (tonic, numpy) at call time,
    so importing this module succeeds even if those packages are not installed.

    Args:
        root: Dataset root directory (will be expanded with os.path.expanduser).
        split: "train" or "test".
        download: Whether to download the dataset if missing.

    Methods:
        load_one(index: int) - returns dataset[index] from tonic.datasets.NMNIST
                               (whatever tonic returns, unmodified).

    Dependency behavior:
    - No error on import of this module when tonic or numpy are not installed.
    - ImportError is raised from load_one(...) with clear install instructions if an optional
      dependency is missing.
    """

    def __init__(self, root: str, split: str = "train", download: bool = False) -> None:
        if split not in ("train", "test"):
            raise ValueError("split must be one of {'train', 'test'}")
        self._root = root
        self._is_train = split == "train"
        self._download = bool(download)

    def load_one(self, index: int) -> Any:
        """
        Load a single sample by index using tonic.datasets.NMNIST.

        This performs lazy imports for 'tonic' and 'numpy' and raises an ImportError with
        actionable guidance if a dependency is missing.

        Returns:
            Whatever tonic.datasets.NMNIST returns for dataset[index].

        Raises:
            ImportError: If 'tonic' or 'numpy' is not available.
        """
        # Lazy-gate optional deps with actionable messages
        tonic = _lazy_import("tonic", "N-MNIST loader")
        _ = _lazy_import("numpy", "N-MNIST loader")

        ds = tonic.datasets.NMNIST(
            save_to=os.path.expanduser(self._root),
            train=self._is_train,
            download=self._download,
        )
        return ds[index]


__all__ = ["NmnistLoader"]