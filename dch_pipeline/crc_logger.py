# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
CRC JSONL logger (opt-in, non-invasive).

Writes one JSON object per line with fields:
{
  "ts": ISO8601 UTC with 'Z',
  "label": str,
  "rule_id": str,
  "type": "Developing" | "Frozen",
  "support": int | float,
  "reliability_mean": float,
  "reliability_ci": [float, float],
  "text": str,
  "provenance_edges": [str]
}

Notes
- Stdlib only. Silent (non-fatal) on exceptions to keep pipeline robust.
- Timestamp is generated at append-time in UTC ISO8601 with 'Z' suffix.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dch_core.crc import CausalRuleCard  # re-use dataclass for typing


def _iso_now_utc() -> str:
    return datetime.utcnow().isoformat() + "Z"


class CRCLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Non-fatal
            pass

    def append(self, card: CausalRuleCard) -> None:
        """
        Append a single JSON line for the provided CausalRuleCard.
        Exceptions are swallowed by design (feature is optional).
        """
        try:
            rec: Dict[str, Any] = {
                "ts": _iso_now_utc(),
                "label": str(card.label),
                "rule_id": str(card.rule_id),
                "type": str(card.type),
                "support": card.support,
                "reliability_mean": float(card.reliability_mean),
                "reliability_ci": [float(card.reliability_ci[0]), float(card.reliability_ci[1])],
                "text": str(card.text),
                "provenance_edges": [str(e) for e in (card.provenance_edges or [])],
            }
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
        except Exception:
            # Non-fatal logging
            pass


__all__ = ["CRCLogger"]