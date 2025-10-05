# dch_core/events.py
"""
Event ingestion, per-neuron ring buffers, and watermark tracking.

Spec alignment:
- Section 1: Formal objects and temporal semantics
- Section 2: DHG TC-kNN requires efficient lookup of the most recent spike
             within [Δ_min, Δ_max] relative to a postsynaptic head time t_j
- Section 9: Data contracts and idempotency
- Section 11: Software blueprint (event lane responsibilities)

This module provides:
- EventStore: orchestrates per-neuron buffers
- EventBuffer: time-sorted append-only buffer with pruning by horizon
- WatermarkTracker: tracks global watermark for ordering / housekeeping

Retention policy:
- Retain events within a configurable time horizon (default 500 ms).
- Prune lazily on ingest and occasionally on explicit calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import bisect

from dch_core.interfaces import (
    EventRecord,
    VertexRecord,
    make_vertex_id,
)

US_PER_MS = 1_000


@dataclass
class BufferConfig:
    """Configuration for per-neuron buffers."""
    horizon_us: int = 500 * US_PER_MS  # retain last 500 ms of spikes
    prune_batch_size: int = 256        # prune after this many appends


class EventBuffer:
    """
    Per-neuron, time-ordered buffer of VertexRecord.
    - Append-only (non-decreasing timestamps are required for correct ordering).
    - Binary search utilities to fetch the most recent event in a window.
    - Lazy pruning by horizon_us.
    """

    __slots__ = ("_ts", "_verts", "_cfg", "_appends")

    def __init__(self, cfg: BufferConfig) -> None:
        self._ts: List[int] = []             # timestamps (us), strictly increasing
        self._verts: List[VertexRecord] = [] # same order as _ts
        self._cfg = cfg
        self._appends = 0

    def __len__(self) -> int:
        return len(self._verts)

    def last_timestamp(self) -> Optional[int]:
        return self._ts[-1] if self._ts else None

    def append(self, v: VertexRecord) -> None:
        """
        Append a vertex (must be in non-decreasing timestamp order).
        If timestamps regress, the buffer will still insert in order,
        but this is considered anomalous for event streams.
        """
        t = v.timestamp_us
        if self._ts and t < self._ts[-1]:
            # Insert to keep sorted (rare: out-of-order event). Use bisect.
            idx = bisect.bisect_right(self._ts, t)
            self._ts.insert(idx, t)
            self._verts.insert(idx, v)
        else:
            self._ts.append(t)
            self._verts.append(v)

        self._appends += 1
        if self._appends >= self._cfg.prune_batch_size:
            self._appends = 0
            self._lazy_prune()

    def _lazy_prune(self) -> None:
        if not self._ts:
            return
        cutoff = self._ts[-1] - self._cfg.horizon_us
        # Find first index with timestamp >= cutoff
        idx = bisect.bisect_left(self._ts, cutoff)
        if idx > 0:
            # Drop everything before idx
            del self._ts[:idx]
            del self._verts[:idx]

    def prune_before(self, min_timestamp_us: int) -> None:
        """
        Force prune all entries strictly before min_timestamp_us.
        """
        idx = bisect.bisect_left(self._ts, min_timestamp_us)
        if idx > 0:
            del self._ts[:idx]
            del self._verts[:idx]

    def most_recent_in_window(
        self,
        head_timestamp_us: int,
        delta_min_us: int,
        delta_max_us: int,
    ) -> Optional[VertexRecord]:
        """
        Return the most recent vertex u in the interval:
            head_timestamp_us - delta_max_us <= u.t < head_timestamp_us - delta_min_us
        If none found, return None.
        """
        if not self._ts:
            return None

        left_bound = head_timestamp_us - delta_max_us
        right_exclusive = head_timestamp_us - delta_min_us

        if right_exclusive <= left_bound:
            return None  # invalid window

        # Rightmost index with t < right_exclusive
        r_idx = bisect.bisect_left(self._ts, right_exclusive) - 1
        if r_idx < 0:
            return None

        # Ensure within left_bound
        if self._ts[r_idx] < left_bound:
            return None

        return self._verts[r_idx]


class WatermarkTracker:
    """
    Tracks a simple watermark for the event stream.

    Definition:
    - watermark_us: last processed event timestamp (monotone non-decreasing).
    Note: In more advanced designs this could be the minimum across multiple
    input queues; here we maintain a single monotone tracker for simplicity.
    """

    __slots__ = ("_watermark_us",)

    def __init__(self) -> None:
        self._watermark_us: int = 0

    def update(self, event_ts_us: int) -> int:
        if event_ts_us > self._watermark_us:
            self._watermark_us = event_ts_us
        return self._watermark_us

    def get(self) -> int:
        return self._watermark_us


class EventStore:
    """
    Manages per-neuron buffers and provides ingestion + window queries.

    Typical usage in the event lane:
    - ingest() each EventRecord to produce a VertexRecord
    - query most_recent_in_window() during DHG TC-kNN candidate search
    """

    __slots__ = ("_buffers", "_cfg", "_wm")

    def __init__(self, cfg: Optional[BufferConfig] = None) -> None:
        self._buffers: Dict[int, EventBuffer] = {}
        self._cfg = cfg or BufferConfig()
        self._wm = WatermarkTracker()

    def get_buffer(self, neuron_id: int) -> EventBuffer:
        buf = self._buffers.get(neuron_id)
        if buf is None:
            buf = EventBuffer(self._cfg)
            self._buffers[neuron_id] = buf
        return buf

    def ingest(self, ev: EventRecord) -> VertexRecord:
        """
        Ingest an EventRecord, create a VertexRecord with stable id,
        append to the neuron's buffer, and advance the watermark.
        """
        v = VertexRecord(
            vertex_id=make_vertex_id(ev.neuron_id, ev.timestamp_us),
            neuron_id=ev.neuron_id,
            timestamp_us=ev.timestamp_us,
        )
        self.get_buffer(ev.neuron_id).append(v)
        self._wm.update(ev.timestamp_us)
        return v

    def watermark(self) -> int:
        """
        Get current watermark_us.
        """
        return self._wm.get()

    def prune_before(self, min_timestamp_us: int) -> None:
        """
        Force prune across all buffers.
        """
        for buf in self._buffers.values():
            buf.prune_before(min_timestamp_us)

    def most_recent_in_window(
        self,
        neuron_id: int,
        head_timestamp_us: int,
        delta_min_us: int,
        delta_max_us: int,
    ) -> Optional[VertexRecord]:
        """
        Wrapper for EventBuffer.most_recent_in_window for a given neuron.
        """
        buf = self._buffers.get(neuron_id)
        if buf is None:
            return None
        return buf.most_recent_in_window(head_timestamp_us, delta_min_us, delta_max_us)


# -----------------------------
# Simple self-test (optional)
# -----------------------------

if __name__ == "__main__":
    store = EventStore()
    # Create a small pattern (Section 1 example)
    a = EventRecord(1, 10_000)
    b = EventRecord(2, 11_700)
    c = EventRecord(3, 21_000)

    va = store.ingest(a)
    vb = store.ingest(b)
    store.ingest(c)

    # Find most recent presyn spikes for head at 21 ms
    u1 = store.most_recent_in_window(1, 21_000, 1_000, 30_000)
    u2 = store.most_recent_in_window(2, 21_000, 1_000, 30_000)

    assert u1 == va, f"Expected va, got {u1}"
    assert u2 == vb, f"Expected vb, got {u2}"
    print("EventStore self-test: OK")