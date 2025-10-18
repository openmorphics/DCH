"""
DCH dataset downloader (torch-free, stdlib-only).

Purpose
- Provide a simple, deterministic CLI to fetch supported event datasets via tonic
  without importing optional dependencies at module import time.

Supported datasets
- nmnist      - tonic.datasets.NMNIST
- dvs_gesture - tonic.datasets.DVSGesture
- all         - convenience to fetch both

Example usage
- python scripts/download_datasets.py --dataset nmnist --root ./data/nmnist --split train
- python scripts/download_datasets.py --dataset dvs_gesture --root ./data/dvs_gesture --split train
- python scripts/download_datasets.py --dataset all --root ./data

Notes
- This script uses lazy imports. If 'tonic' is not installed, it emits a clear, actionable
  error with pip/conda-forge instructions and exits with code 1.
- No external dependencies beyond Python stdlib are required to run this CLI.

Exit behavior
- On success: prints a single JSON line and exits 0
- On error:   prints a single JSON line with {"error": "..."} and exits 1
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any, Dict, Optional


def _json_out(payload: Dict[str, Any], ok: bool = True) -> None:
    """
    Print a single JSON object line to stdout and exit.
    ok=True exits with code 0; else 1.
    """
    try:
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    except Exception as e:
        # Fallback in pathological cases
        print('{"error":"failed to serialize JSON output"}')
        sys.exit(1)
    sys.exit(0 if ok else 1)


def _lazy_import(name: str, purpose: str) -> Any:
    """
    Lazy import helper with actionable guidance.

    Args:
        name: module name to import (e.g., 'tonic')
        purpose: human-readable purpose for error messaging
    """
    try:
        return importlib.import_module(name)
    except Exception as e:
        _json_out(
            {
                "error": f"Optional dependency '{name}' is required for {purpose}.",
                "hint": {
                    "pip": f"pip install {name}",
                    "conda-forge": f"conda install -c conda-forge {name}",
                },
            },
            ok=False,
        )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_root(dataset: str, root: str) -> str:
    """
    Normalize root path; if --dataset all and a shared --root was provided,
    we create subfolders under root per dataset.
    """
    root = os.path.abspath(root)
    return root


def _download_nmnist(root: str, split: str) -> Dict[str, Any]:
    tonic = _lazy_import("tonic", "downloading N-MNIST")
    ds_mod = getattr(tonic, "datasets", None)
    if ds_mod is None:
        _json_out({"error": "tonic.datasets not found"}, ok=False)

    cls = getattr(ds_mod, "NMNIST", None)
    if cls is None:
        _json_out({"error": "tonic.datasets.NMNIST missing in installed tonic"}, ok=False)

    train = True if split == "train" else False
    _ensure_dir(root)

    # Many tonic datasets accept 'save_to', 'train', 'download' flags
    ds = cls(save_to=root, train=train, download=True)  # type: ignore[call-arg]
    # Touch a small access to ensure materialization side effects are likely performed
    _ = len(ds)  # type: ignore[arg-type]

    return {
        "dataset": "nmnist",
        "root": root,
        "split": split,
        "status": "ok",
        "count": int(_),
    }


def _download_dvs_gesture(root: str, split: str) -> Dict[str, Any]:
    tonic = _lazy_import("tonic", "downloading DVS Gesture")
    ds_mod = getattr(tonic, "datasets", None)
    if ds_mod is None:
        _json_out({"error": "tonic.datasets not found"}, ok=False)

    cls = getattr(ds_mod, "DVSGesture", None)
    if cls is None:
        _json_out({"error": "tonic.datasets.DVSGesture missing in installed tonic"}, ok=False)

    train = True if split == "train" else False
    _ensure_dir(root)

    ds = cls(save_to=root, train=train, download=True)  # type: ignore[call-arg]
    _ = len(ds)  # type: ignore[arg-type]

    return {
        "dataset": "dvs_gesture",
        "root": root,
        "split": split,
        "status": "ok",
        "count": int(_),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DCH dataset downloader (lazy optional-deps, torch-free)."
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=["nmnist", "dvs_gesture", "all"],
        help="Dataset to download.",
    )
    p.add_argument(
        "--root",
        required=True,
        help="Root directory to store dataset (for 'all', acts as a parent directory).",
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Split to download (if dataset supports splits).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity; output remains a single JSON line on success/failure.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    try:
        if args.dataset == "nmnist":
            root = _normalize_root("nmnist", args.root)
            result = _download_nmnist(root, args.split)
            _json_out(result, ok=True)

        elif args.dataset == "dvs_gesture":
            root = _normalize_root("dvs_gesture", args.root)
            result = _download_dvs_gesture(root, args.split)
            _json_out(result, ok=True)

        elif args.dataset == "all":
            # For 'all', create subfolders under the provided root
            parent = _normalize_root("all", args.root)
            nmnist_root = os.path.join(parent, "nmnist")
            dvs_root = os.path.join(parent, "dvs_gesture")
            res1 = _download_nmnist(nmnist_root, args.split)
            res2 = _download_dvs_gesture(dvs_root, args.split)
            _json_out(
                {
                    "status": "ok",
                    "datasets": [res1, res2],
                },
                ok=True,
            )
        else:
            _json_out({"error": f"unsupported dataset '{args.dataset}'"}, ok=False)

    except SystemExit:
        raise
    except Exception as e:
        _json_out({"error": f"{e.__class__.__name__}: {str(e)}"}, ok=False)


if __name__ == "__main__":
    main()