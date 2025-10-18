# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Episode Audit Trails (EAT) JSONL logger with SHA-256 hash chaining.

Schema (one record per line):
{
  "ts": ISO8601 string (UTC, suffix 'Z'),
  "kind": "EAT" | "GROW" | "UPDATE",
  "payload": { ... },  # see below
  "prev_hash": "hex" | null,
  "hash": "hex"
}

Hashing:
- Let R_nohash be the JSON object without the "hash" field and with "prev_hash" set
  to the previous record's hash as a hex string (or null for the first record).
- Let bytes_json = UTF-8 encoding of json.dumps(R_nohash, separators=(',', ':'), sort_keys=True)
- Let prev_bytes = bytes.fromhex(prev_hash) if prev_hash is not None else b""
- Then: hash_hex = SHA256(bytes_json + prev_bytes).hexdigest()
- This hash chaining ensures tamper-evident append-only logs.

Payloads:
- EAT payload (for Hyperpaths):
  {
    "head": str,
    "edges": [str],
    "score": float,
    "length": int,
    "label": str | null
  }
- GROW payload:
  {
    "admitted": [str],
    "count": int
  }
- UPDATE payload:
  {
    "edges": [{"id": str, "reliability": float}],
    "count": int
  }

Timestamps:
- 'ts' is always set to current UTC time in ISO8601 with trailing 'Z' unless provided by caller
  (emit_grow/emit_update accept explicit now_t_iso to align with pipeline step time).

Concurrency:
- This implementation assumes single-process append access (P1 scope). For multi-process safety,
  one would need atomic append or file locks (platform-dependent). Left for future work.

Verification:
- verify() performs a full-file hash-chain integrity check and returns:
    {"ok": bool, "count": n, "bad_index": i | None}
  where bad_index is the zero-based index of the first invalid record.

References:
- Hyperpath dataclass: see dch_core.interfaces.Hyperpath
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from dch_core.interfaces import Hyperpath  # type: ignore


def _now_iso_utc() -> str:
    """Return ISO8601 UTC string with 'Z' suffix."""
    # Keep simple formatting; tests only check existence/consistency
    return datetime.utcnow().isoformat() + "Z"


class EATAuditLogger:
    """
    Tamper-evident JSONL logger for Episode Audit Trails with SHA-256 hash chaining.

    Usage:
    - Construct once per run with the target path. Existing log is resumed by reading
      the last line to recover the trailing hash, allowing append-safe continuation.
    - Call emit_grow/emit_eat/emit_update from the pipeline hooks. Emission failures
      are non-fatal and are swallowed (P1 scope).

    Parameters
    ----------
    path : str
        Target JSONL file path.
    retention_max_bytes : int | None
        Optional retention soft-limit. When file size exceeds this number of bytes,
        the current file is rotated (renamed with a UTC suffix) and a fresh file is opened.
        The hash chain restarts after rotation (first record has prev_hash = null).
    """

    def __init__(self, path: str, retention_max_bytes: Optional[int] = None) -> None:
        self.path = Path(path)
        self.retention_max_bytes = int(retention_max_bytes) if retention_max_bytes else None

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", encoding="utf-8")
        self._last_hash: Optional[bytes] = None

        # Resume from existing log tail
        try:
            if self.path.exists() and self.path.stat().st_size > 0:
                last_line = self._read_last_nonempty_line()
                if last_line:
                    rec = json.loads(last_line)
                    h = rec.get("hash")
                    if isinstance(h, str) and len(h) > 0:
                        self._last_hash = bytes.fromhex(h)
        except Exception:
            # Non-fatal; start new chain
            self._last_hash = None

    # ------------- Public API -------------

    def emit_event(self, record: Dict[str, Any]) -> None:
        """
        Generic append of a pre-formed record with keys: ts, kind, payload.

        This method enriches with 'prev_hash' and 'hash' and appends to the file.
        Exceptions are swallowed to keep pipeline robust.
        """
        try:
            self.rotate_if_needed()

            base: Dict[str, Any] = {}
            # Normalize minimal schema
            ts = record.get("ts") or _now_iso_utc()
            base["ts"] = ts
            base["kind"] = record["kind"]
            base["payload"] = record.get("payload", {})
            prev_hex = self._last_hash.hex() if self._last_hash is not None else None
            base["prev_hash"] = prev_hex

            # Compute chain hash on canonical JSON (without 'hash') + prev bytes
            data_bytes = json.dumps(base, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
            h = hashlib.sha256()
            h.update(data_bytes)
            if self._last_hash is not None:
                h.update(self._last_hash)
            digest_hex = h.hexdigest()

            # Finalize record and write
            base["hash"] = digest_hex
            self._f.write(json.dumps(base, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
            self._f.flush()
            self._last_hash = bytes.fromhex(digest_hex)
        except Exception:
            # P1: non-fatal side-effect; ignore
            pass

    def emit_eat(self, hyperpath: Hyperpath, now_t_us: int, meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Emit an EAT record for a given Hyperpath.

        Payload:
        - head: str(hyperpath.head)
        - edges: [str(e) for e in hyperpath.edges]
        - score: float(hyperpath.score)
        - length: int(hyperpath.length)
        - label: hyperpath.label (or None)

        Parameters
        ----------
        hyperpath : Hyperpath
            The discovered hyperpath instance (see dch_core.interfaces.Hyperpath).
        now_t_us : int
            Event-time in microseconds (not serialized; reserved for future use).
        meta : dict | None
            Reserved for future metadata (ignored in P1 schema).
        """
        try:
            payload = {
                "head": str(hyperpath.head),
                "edges": [str(e) for e in hyperpath.edges],
                "score": float(hyperpath.score),
                "length": int(hyperpath.length),
                "label": hyperpath.label if (hyperpath.label is None or isinstance(hyperpath.label, str)) else str(hyperpath.label),
            }
            self.emit_event({"ts": _now_iso_utc(), "kind": "EAT", "payload": payload})
        except Exception:
            pass

    def emit_grow(self, admitted_eids: List[str], now_t_iso: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Emit a GROW record listing admitted edge ids.

        Parameters
        ----------
        admitted_eids : list[str]
            Admitted edge identifiers serialized as strings.
        now_t_iso : str
            ISO8601 UTC timestamp for alignment with pipeline step time.
        meta : dict | None
            Reserved (ignored in P1 schema).
        """
        try:
            payload = {"admitted": [str(e) for e in admitted_eids], "count": int(len(admitted_eids))}
            self.emit_event({"ts": now_t_iso or _now_iso_utc(), "kind": "GROW", "payload": payload})
        except Exception:
            pass

    def emit_update(self, updated: Mapping[str, float], now_t_iso: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Emit an UPDATE record of reliability changes.

        Parameters
        ----------
        updated : dict[str, float]
            Mapping of edge id to reliability.
        now_t_iso : str
            ISO8601 UTC timestamp for alignment with pipeline step time.
        meta : dict | None
            Reserved (ignored in P1 schema).
        """
        try:
            edges = [{"id": str(k), "reliability": float(v)} for k, v in updated.items()]
            payload = {"edges": edges, "count": int(len(edges))}
            self.emit_event({"ts": now_t_iso or _now_iso_utc(), "kind": "UPDATE", "payload": payload})
        except Exception:
            pass

    def rotate_if_needed(self) -> None:
        """
        Rotate the current file if retention_max_bytes is exceeded.

        Rotation renames the active file to <name>.<UTC_ISO_COMPACT> and starts a fresh chain.
        """
        if not self.retention_max_bytes:
            return
        try:
            size = self.path.stat().st_size if self.path.exists() else 0
            if size > self.retention_max_bytes:
                # Close current
                try:
                    self._f.flush()
                except Exception:
                    pass
                try:
                    self._f.close()
                except Exception:
                    pass

                suffix = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                rotated = self.path.with_name(f"{self.path.name}.{suffix}")
                os.replace(self.path, rotated)

                # Reopen fresh and reset chain head
                self._f = self.path.open("a", encoding="utf-8")
                self._last_hash = None
        except Exception:
            # Non-fatal
            pass

    def verify(self) -> Dict[str, Any]:
        """Full-file hash-chain integrity verification for this logger's file."""
        return verify_file(str(self.path))

    # ------------- Helpers -------------

    def _read_last_nonempty_line(self) -> Optional[str]:
        try:
            with self.path.open("r", encoding="utf-8") as f:
                last: Optional[str] = None
                for line in f:
                    s = line.strip()
                    if s:
                        last = s
                return last
        except Exception:
            return None


def verify_file(path: str) -> Dict[str, Any]:
    """
    Verify the integrity of an EAT JSONL file created by EATAuditLogger.

    Returns
    -------
    dict
        {"ok": bool, "count": int, "bad_index": int | None}
        - ok: True if the entire chain is valid
        - count: number of valid lines processed
        - bad_index: zero-based index of the first invalid record, else None
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {"ok": True, "count": 0, "bad_index": None}

    last_hash: Optional[bytes] = None
    count = 0

    try:
        with p.open("r", encoding="utf-8") as f:
            for idx, raw in enumerate(f):
                s = raw.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except Exception:
                    return {"ok": False, "count": count, "bad_index": idx}

                # Basic schema presence
                if not isinstance(rec, dict) or "ts" not in rec or "kind" not in rec or "payload" not in rec or "hash" not in rec:
                    return {"ok": False, "count": count, "bad_index": idx}

                prev_hex_in_file = rec.get("prev_hash", None)
                expected_prev_hex = last_hash.hex() if last_hash is not None else None
                if prev_hex_in_file != expected_prev_hex:
                    return {"ok": False, "count": count, "bad_index": idx}

                # Recompute hash from record without 'hash'
                rec_nohash = dict(rec)
                rec_nohash.pop("hash", None)
                data_bytes = json.dumps(rec_nohash, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
                h = hashlib.sha256()
                h.update(data_bytes)
                if last_hash is not None:
                    h.update(last_hash)
                recomputed = h.hexdigest()

                if recomputed != rec.get("hash"):
                    return {"ok": False, "count": count, "bad_index": idx}

                last_hash = bytes.fromhex(recomputed)
                count += 1

        return {"ok": True, "count": count, "bad_index": None}
    except Exception:
        return {"ok": False, "count": count, "bad_index": count}