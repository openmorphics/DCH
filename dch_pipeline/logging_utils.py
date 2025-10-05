# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Logging and artifact management utilities for DCH.

This module is intentionally torch-free and filesystem-only by default.
TensorBoard support is included behind lazy optional imports.

Exports:
- CSVLogger: append-safe CSV writer with header-once semantics.
- JSONLLogger: newline-delimited JSON writer.
- TensorBoardLogger: optional TensorBoard writer (no-op if unavailable).
- ExperimentLogger: composite facade over CSV/JSONL/TensorBoard.
- JsonIO: tiny JSON helpers.
- make_run_dir: create a timestamped run directory for artifacts.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union


# -------------------------
# Filesystem helpers
# -------------------------


def ensure_dir(path: Path) -> None:
    """Create the directory if it does not already exist (parents included)."""
    Path(path).mkdir(parents=True, exist_ok=True)


def timestamp_id() -> str:
    """Return a compact UTC timestamp suitable for folder names."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S")


def make_run_dir(root: Path, *, prefix: Optional[str] = None) -> Path:
    """
    Create a new run directory under 'root' with a UTC timestamp id.

    Example:
        root=artifacts, prefix=None => artifacts/20251004T201500/
        root=artifacts, prefix=exp1 => artifacts/exp1_20251004T201500/
    """
    ensure_dir(root)
    tid = timestamp_id()
    name = f"{prefix}_{tid}" if prefix else tid
    d = root / name
    ensure_dir(d)
    return d


# -------------------------
# CSV metrics logger
# -------------------------


class CSVLogger:
    """
    Append-safe CSV writer that writes the header only once.

    Parameters
    ----------
    path : str | Path
        Target CSV file path.
    fieldnames : list[str] | None
        If None, infer from the first row's keys sorted lexicographically.
        If provided, enforce consistent field order.
    write_header : bool
        Whether to write a header (if the file is new/empty).
    allow_extra : bool
        If False, extra keys in rows raise ValueError. If True, extras are ignored.
    """

    def __init__(
        self,
        path: Union[str, Path],
        fieldnames: Optional[Sequence[str]] = None,
        write_header: bool = True,
        allow_extra: bool = False,
    ) -> None:
        self.path = Path(path)
        self._fieldnames: Optional[list[str]] = list(fieldnames) if fieldnames is not None else None
        self._write_header = bool(write_header)
        self._allow_extra = bool(allow_extra)

        ensure_dir(self.path.parent)

        # If file already exists and is non-empty, we assume header is present.
        self._header_written = (self.path.exists() and self.path.stat().st_size > 0) or not self._write_header

        self._f = None  # Will be a TextIO object opened lazily in append mode
        self._writer: Optional[csv.DictWriter] = None

    # Back-compat convenience
    def write(self, row: Mapping[str, Any]) -> None:
        """Alias for write_row."""
        self.write_row(row)

    def _ensure_open(self) -> None:
        if self._f is None:
            self._f = self.path.open("a", newline="", encoding="utf-8")

    def _ensure_writer(self, row: Optional[Mapping[str, Any]] = None) -> None:
        if self._writer is not None:
            return
        if self._fieldnames is None:
            if row is None:
                raise ValueError("fieldnames not provided and cannot infer from empty input.")
            # Deterministic order: lexicographically sorted keys
            self._fieldnames = sorted(row.keys())

        extrasaction = "ignore" if self._allow_extra else "raise"
        self._writer = csv.DictWriter(
            self._f,
            fieldnames=self._fieldnames,
            extrasaction=extrasaction,
            restval="",
        )
        if not self._header_written and self._write_header:
            self._writer.writeheader()
            self._header_written = True
            # ensure durability of header in case of early program exit
            self.flush()

    def write_row(self, row: Mapping[str, Any]) -> None:
        """Write a single row mapping."""
        self._ensure_open()
        self._ensure_writer(row)
        assert self._writer is not None
        self._writer.writerow(row)

    def write_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Write multiple row mappings."""
        for r in rows:
            self.write_row(r)

    def flush(self) -> None:
        """Flush the underlying file buffer."""
        if self._f is not None:
            self._f.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        try:
            if self._f is not None:
                self._f.flush()
                self._f.close()
        finally:
            self._f = None
            self._writer = None

    # Context manager support
    def __enter__(self) -> "CSVLogger":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# -------------------------
# JSONL (newline-delimited JSON) logger
# -------------------------


class JSONLLogger:
    """
    Newline-delimited JSON writer with append semantics.

    Parameters
    ----------
    path : str | Path
        Target .jsonl file path.
    auto_timestamp : bool
        If True, inject a 'ts' ISO8601 string when not present in the record.
    """

    def __init__(self, path: Union[str, Path], auto_timestamp: bool = False) -> None:
        self.path = Path(path)
        self.auto_timestamp = bool(auto_timestamp)
        ensure_dir(self.path.parent)
        self._f = self.path.open("a", encoding="utf-8")

    def log(self, record: Mapping[str, Any]) -> None:
        """Append a single JSON record as one line."""
        data = dict(record)
        if self.auto_timestamp and "ts" not in data:
            # Minimal ISO8601 representation; format validation is not required.
            data["ts"] = datetime.utcnow().isoformat() + "Z"
        self._f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        if self._f is not None:
            self._f.flush()

    def close(self) -> None:
        try:
            if self._f is not None:
                self._f.flush()
                self._f.close()
        finally:
            self._f = None

    def __enter__(self) -> "JSONLLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# -------------------------
# Lazy import helper
# -------------------------


def _lazy_import(name: str, purpose: str) -> Any:
    """
    Attempt to import a module by name, raising ImportError with helpful hints.

    Parameters
    ----------
    name : str
        The module path to import (e.g., 'torch.utils.tensorboard' or 'tensorboardX').
    purpose : str
        Human-readable description of what the import is used for (e.g., 'TensorBoard logging').

    Returns
    -------
    module
        The imported module on success.

    Raises
    ------
    ImportError
        With actionable pip/conda-forge hints.
    """
    try:
        return import_module(name)
    except Exception as e:
        hint = (
            f"Optional dependency missing for {purpose!r}: {name}\n"
            "- Try: pip install tensorboard\n"
            "- Or:  pip install tensorboardX\n"
            "- Conda (conda-forge): conda install -c conda-forge tensorboard tensorboardx\n"
        )
        raise ImportError(hint) from e


# -------------------------
# Optional TensorBoard logger (no-op if backend missing)
# -------------------------


class TensorBoardLogger:
    """
    Lazy, optional TensorBoard logger.

    - Tries 'torch.utils.tensorboard.SummaryWriter' first, then 'tensorboardX.SummaryWriter'.
    - If unavailable, becomes a no-op with the same API, and does not create directories.

    Methods
    -------
    log_scalar(tag, value, step)
    log_scalars(tag, dict, step)
    log_histogram(tag, values, step)
    flush()
    close()

    Attributes
    ----------
    is_active : bool
        True if a real backend is available and instantiated.
    """

    def __init__(self, log_dir: Union[str, Path]) -> None:
        self.log_dir = Path(log_dir)
        self._writer = None  # SummaryWriter instance if available
        self._init_backend()

    # Internal: resolve and create SummaryWriter if available
    def _init_backend(self) -> None:
        SummaryWriter = None
        for mod_name in ("torch.utils.tensorboard", "tensorboardX"):
            try:
                mod = import_module(mod_name)
            except Exception:
                continue
            SummaryWriter = getattr(mod, "SummaryWriter", None)
            if SummaryWriter is not None:
                break

        if SummaryWriter is not None:
            try:
                ensure_dir(self.log_dir)
                self._writer = SummaryWriter(log_dir=str(self.log_dir))
            except Exception:
                # If backend instantiation fails, remain no-op
                self._writer = None

    @property
    def is_active(self) -> bool:
        """Whether a real backend is active."""
        return self._writer is not None

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        if not self.is_active:
            return
        try:
            self._writer.add_scalar(tag, float(value), None if step is None else int(step))
        except Exception:
            pass

    def log_scalars(self, tag: str, values: Mapping[str, float], step: Optional[int] = None) -> None:
        if not self.is_active:
            return
        # iteratively forward to add_scalar
        for k, v in values.items():
            self.log_scalar(f"{tag}/{k}", v, step)

    def log_histogram(self, tag: str, values: Any, step: Optional[int] = None) -> None:
        if not self.is_active:
            return
        try:
            # We pass values through without heavy processing (torch-free).
            self._writer.add_histogram(tag, values, None if step is None else int(step))
        except Exception:
            pass

    def flush(self) -> None:
        if self.is_active:
            try:
                self._writer.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self.is_active:
            try:
                self._writer.flush()
                self._writer.close()
            except Exception:
                pass


# Backwards-compatible alias
TBLogger = TensorBoardLogger


# -------------------------
# Composite Experiment Logger
# -------------------------


class ExperimentLogger:
    """
    Thin facade that can wrap multiple loggers (CSV, JSONL, TensorBoard).

    Parameters
    ----------
    csv_path : str | Path | None
        If provided, instantiate a CSVLogger at this path.
    csv_fieldnames : list[str] | None
        Optional explicit field order for CSVLogger.
    jsonl_path : str | Path | None
        If provided, instantiate a JSONLLogger at this path (without auto-timestamp).
    tb_log_dir : str | Path | None
        If provided, instantiate a TensorBoardLogger at this directory.
    """

    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        csv_fieldnames: Optional[Sequence[str]] = None,
        jsonl_path: Optional[Union[str, Path]] = None,
        tb_log_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.csv: Optional[CSVLogger] = CSVLogger(csv_path, fieldnames=csv_fieldnames) if csv_path else None
        self.jsonl: Optional[JSONLLogger] = JSONLLogger(jsonl_path) if jsonl_path else None
        self.tb: Optional[TensorBoardLogger] = TensorBoardLogger(tb_log_dir) if tb_log_dir else None

    # CSV
    def log_csv(self, row: Mapping[str, Any]) -> None:
        if self.csv is not None:
            self.csv.write_row(row)

    def log_csv_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        if self.csv is not None:
            self.csv.write_rows(rows)

    # JSONL
    def log_jsonl(self, record: Mapping[str, Any]) -> None:
        if self.jsonl is not None:
            self.jsonl.log(record)

    # TensorBoard
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        if self.tb is not None:
            self.tb.log_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: Mapping[str, float], step: Optional[int] = None) -> None:
        if self.tb is not None:
            self.tb.log_scalars(tag, values, step)

    def log_histogram(self, tag: str, values: Any, step: Optional[int] = None) -> None:
        if self.tb is not None:
            self.tb.log_histogram(tag, values, step)

    # Lifecycle
    def flush(self) -> None:
        if self.csv is not None:
            self.csv.flush()
        if self.jsonl is not None:
            self.jsonl.flush()
        if self.tb is not None:
            self.tb.flush()

    def close(self) -> None:
        if self.csv is not None:
            self.csv.close()
        if self.jsonl is not None:
            self.jsonl.close()
        if self.tb is not None:
            self.tb.close()


# -------------------------
# JSON helpers (kept for convenience/back-compat)
# -------------------------


class JsonIO:
    """Tiny JSON writer/reader helpers with UTF-8 and pretty-printing."""

    @staticmethod
    def write(path: Path, obj: Any, *, sort_keys: bool = True, indent: int = 2) -> None:
        ensure_dir(path.parent)
        if is_dataclass(obj):
            obj = asdict(obj)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, sort_keys=sort_keys)

    @staticmethod
    def read(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)


__all__ = [
    "CSVLogger",
    "JSONLLogger",
    "TensorBoardLogger",
    "ExperimentLogger",
    "TBLogger",
    "JsonIO",
    "make_run_dir",
    "ensure_dir",
    "timestamp_id",
    "_lazy_import",
]