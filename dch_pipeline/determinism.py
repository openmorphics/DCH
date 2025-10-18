"""Determinism utilities.

Provides:
- load_first_jsonl_record(path): Load the first JSON object from a JSONL file.
- _float_equal(a, b, tol): Absolute tolerance comparison.
- compare_artifacts(a, b, tol=0.0): Compare artifact dictionaries for deterministic equality with optional numeric tolerance.
- make_manifest(records, comparisons=None, label=None): Produce a concise manifest for a results bundle.

Notes
- This module is stdlib-only and does not alter any existing protocols.
- Artifacts are expected to contain at least:
    {"config_fingerprint": str, "metrics": dict, "reliability_summary": {"mean_reliability": float, "edges": [{"id": str?, "mean": float, "ci": [lo, hi]}]}}
- Extra keys are ignored during comparisons unless specifically handled below.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, List, Dict, Tuple, Optional
import json
import os


def load_first_jsonl_record(path: str) -> dict:
    """
    Load and return the first JSON object from a JSONL file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty, the first non-empty line is not valid JSON,
                   or the record is not a JSON object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Expected a file path, got directory: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in first JSONL line: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError("First JSONL record is not a JSON object")
            return obj

    raise ValueError(f"Empty JSONL file: {path}")


def _float_equal(a: float, b: float, tol: float) -> bool:
    """Return True iff absolute difference <= tol."""
    try:
        return abs(float(a) - float(b)) <= float(tol)
    except Exception:
        return False


def _is_number(x: Any) -> bool:
    # bool is a subclass of int; treat bools separately
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _edge_sort_key(e: Mapping[str, Any]) -> Tuple[str, Any]:
    if "id" in e:
        try:
            return ("id", str(e.get("id")))
        except Exception:
            pass
    # fallback sort by mean
    try:
        return ("mean", float(e.get("mean", 0.0)))
    except Exception:
        return ("mean", 0.0)


def compare_artifacts(a: Mapping[str, Any], b: Mapping[str, Any], *, tol: float = 0.0) -> dict:
    """
    Compare two artifact dictionaries with tolerant numeric comparison.

    Compared sections:
    - config_fingerprint: exact string equality (reported as a boolean in 'diff').
    - metrics: only common keys are compared.
        * numeric scalars: absolute difference <= tol considered equal
        * strings/bools: exact equality
        * other types are ignored
    - reliability_summary:
        * mean_reliability: within tol
        * edges: compare len and pairwise elements after order-insensitive sort
                 (by 'id' if present else by 'mean'); 'mean' within tol and 'ci'
                 element-wise within tol.

    Returns:
        {
          "equal": bool,
          "diff": {
            "config_fingerprint": bool,  # True if equal, False if not
            "metrics": {key: {"a": v1, "b": v2}},  # only keys that differ
            "reliability_summary": {
                "mean": {"a": m1, "b": m2},                # present only if differ
                "edges_length": {"a": la, "b": lb},        # present only if differ
                "edges": [                                  # present only if any pair differs
                    {
                      "index": i,
                      "id_a": edge_id_a or None,
                      "id_b": edge_id_b or None,
                      "mean": {"a": ma, "b": mb},          # present if mean differs
                      "ci": {"a": [loa, hia], "b": [lob, hib]}  # present if any CI bound differs
                    }, ...
                ]
            }
          }
        }
    """
    diff: Dict[str, Any] = {}
    # 1) config_fingerprint
    cf_equal = (a.get("config_fingerprint") == b.get("config_fingerprint"))
    diff["config_fingerprint"] = bool(cf_equal)

    # 2) metrics diffs (only common keys)
    mdiff: Dict[str, Any] = {}
    ma = a.get("metrics") or {}
    mb = b.get("metrics") or {}
    if isinstance(ma, Mapping) and isinstance(mb, Mapping):
        common_keys = set(ma.keys()) & set(mb.keys())
        for k in sorted(common_keys):
            va = ma.get(k)
            vb = mb.get(k)
            if _is_number(va) and _is_number(vb):
                if not _float_equal(float(va), float(vb), tol):
                    mdiff[k] = {"a": va, "b": vb}
            elif isinstance(va, (str, bool)) and isinstance(vb, type(va)):
                if va != vb:
                    mdiff[k] = {"a": va, "b": vb}
            else:
                # unspecified types: ignore
                continue
    else:
        # Structure mismatch counts as a difference
        mdiff["__structure__"] = {"a": type(ma).__name__, "b": type(mb).__name__}
    diff["metrics"] = mdiff

    # 3) reliability_summary diffs
    rs_diff: Dict[str, Any] = {}
    ra = a.get("reliability_summary") or {}
    rb = b.get("reliability_summary") or {}
    if not isinstance(ra, Mapping) or not isinstance(rb, Mapping):
        rs_diff["__structure__"] = {"a": type(ra).__name__, "b": type(rb).__name__}
    else:
        # mean_reliability
        try:
            ma_mean = float(ra.get("mean_reliability", 0.0))
        except Exception:
            ma_mean = 0.0
        try:
            mb_mean = float(rb.get("mean_reliability", 0.0))
        except Exception:
            mb_mean = 0.0
        if not _float_equal(ma_mean, mb_mean, tol):
            rs_diff["mean"] = {"a": ma_mean, "b": mb_mean}

        # edges list
        ea = ra.get("edges") or []
        eb = rb.get("edges") or []
        if not isinstance(ea, list) or not isinstance(eb, list):
            rs_diff["edges_structure"] = {"a": type(ea).__name__, "b": type(eb).__name__}
        else:
            la = len(ea)
            lb = len(eb)
            if la != lb:
                rs_diff["edges_length"] = {"a": la, "b": lb}
            # Sort and compare pairwise up to min length
            sa = sorted([e for e in ea if isinstance(e, Mapping)], key=_edge_sort_key)
            sb = sorted([e for e in eb if isinstance(e, Mapping)], key=_edge_sort_key)
            n = min(len(sa), len(sb))
            edge_diffs: List[Dict[str, Any]] = []
            for i in range(n):
                ea_i = sa[i]
                eb_i = sb[i]
                ed: Dict[str, Any] = {"index": i, "id_a": ea_i.get("id"), "id_b": eb_i.get("id")}
                # mean
                try:
                    mea = float(ea_i.get("mean", 0.0))
                except Exception:
                    mea = 0.0
                try:
                    meb = float(eb_i.get("mean", 0.0))
                except Exception:
                    meb = 0.0
                if not _float_equal(mea, meb, tol):
                    ed["mean"] = {"a": mea, "b": meb}
                # ci
                cia = ea_i.get("ci")
                cib = eb_i.get("ci")
                ci_diff = False
                if (
                    isinstance(cia, (list, tuple)) and len(cia) == 2
                    and isinstance(cib, (list, tuple)) and len(cib) == 2
                ):
                    try:
                        loa, hia = float(cia[0]), float(cia[1])
                        lob, hib = float(cib[0]), float(cib[1])
                        if (not _float_equal(loa, lob, tol)) or (not _float_equal(hia, hib, tol)):
                            ci_diff = True
                            ed["ci"] = {"a": [loa, hia], "b": [lob, hib]}
                    except Exception:
                        # if cannot parse floats, treat as difference
                        ci_diff = True
                        ed["ci"] = {"a": cia, "b": cib}
                elif cia != cib:
                    ci_diff = True
                    ed["ci"] = {"a": cia, "b": cib}

                if ("mean" in ed) or ("ci" in ed):
                    edge_diffs.append(ed)

            if edge_diffs:
                rs_diff["edges"] = edge_diffs

    diff["reliability_summary"] = rs_diff

    equal = bool(cf_equal) and len(mdiff) == 0 and len(rs_diff) == 0
    return {"equal": equal, "diff": diff}


def make_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    comparisons: Sequence[dict] | None = None,
    label: str | None = None,
) -> dict:
    """
    Build a concise manifest for a results bundle.
    """
    cfg_fps: List[str] = []
    environments: List[Mapping[str, Any]] = []
    for rec in records:
        cf = rec.get("config_fingerprint")
        if isinstance(cf, str) and cf:
            cfg_fps.append(cf)
        if "env" in rec:
            environments.append(rec.get("env") or {})

    manifest = {
        "label": label or "results_bundle",
        "n": len(records),
        "config_fingerprints": cfg_fps,
        "environments": environments,
        "comparisons": list(comparisons or []),
    }
    return manifest


__all__ = ["load_first_jsonl_record", "compare_artifacts", "make_manifest"]