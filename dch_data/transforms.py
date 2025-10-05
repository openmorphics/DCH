# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Event preprocessing transforms for DCH.

These utilities operate purely on the lightweight Event dataclass
([Event](dch_core/interfaces.py:71)) without requiring torch or dataset-specific
dependencies. They are designed to compose functionally and keep the pipeline
CPU-friendly by default.

Provided transforms:
- [time_window()](dch_data/transforms.py:35): filter events by [t0, t1] inclusive window
- [neuron_filter()](dch_data/transforms.py:58): include/exclude by neuron id
- [time_normalize()](dch_data/transforms.py:90): shift timestamps to start at zero or a provided origin
- [sort_events()](dch_data/transforms.py:113): sort by (t, neuron_id)
- [subsample()](dch_data/transforms.py:123): take every k-th event in time order
- [chunk_by_bins()](dch_data/transforms.py:146): group events into fixed-width time bins within a window
- [sliding_windows()](dch_data/transforms.py:187): produce overlapping temporal windows of events
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from dch_core.interfaces import Event, Window, Timestamp, NeuronId


def time_window(events: Sequence[Event], window: Window) -> List[Event]:
    """
    Filter events to timestamps within [t0, t1] inclusive.

    Args:
        events: sequence of Event
        window: (t0, t1) inclusive

    Returns:
        New list of events with t in [t0, t1].
    """
    t0, t1 = window
    if t1 < t0:
        return []
    return [e for e in events if t0 <= e.t <= t1]


def neuron_filter(
    events: Sequence[Event],
    *,
    allowlist: Optional[Iterable[NeuronId]] = None,
    denylist: Optional[Iterable[NeuronId]] = None,
) -> List[Event]:
    """
    Keep or remove events based on neuron ids.

    Precedence:
        - If allowlist provided: keep only those neuron ids (denylist applied after).
        - If denylist provided: remove those neuron ids.

    Returns:
        Filtered list of events.
    """
    allow: Optional[set[int]] = set(int(n) for n in allowlist) if allowlist is not None else None
    deny: Optional[set[int]] = set(int(n) for n in denylist) if denylist is not None else None

    out: List[Event] = []
    for e in events:
        nid = int(e.neuron_id)
        if allow is not None and nid not in allow:
            continue
        if deny is not None and nid in deny:
            continue
        out.append(e)
    return out


def time_normalize(
    events: Sequence[Event],
    *,
    t0: Optional[Timestamp] = None,
) -> List[Event]:
    """
    Shift event timestamps so that the earliest time maps to 0 (or provided origin).

    Args:
        t0: when None, uses min(e.t) from input; otherwise shifts by provided t0.

    Returns:
        New list of Event with updated timestamps; preserves 'meta' mapping.
    """
    if not events:
        return []
    origin = int(t0) if t0 is not None else int(min(e.t for e in events))
    out: List[Event] = [Event(neuron_id=e.neuron_id, t=int(e.t - origin), meta=e.meta) for e in events]
    return out


def sort_events(events: Sequence[Event]) -> List[Event]:
    """
    Sort events by (timestamp, neuron_id) to ensure deterministic order.
    """
    return sorted(events, key=lambda e: (int(e.t), int(e.neuron_id)))


def subsample(events: Sequence[Event], *, stride: int = 1) -> List[Event]:
    """
    Return every k-th event in time order (after sorting).

    Args:
        stride: positive integer; 1 returns all events.

    Returns:
        Subsampled list of events.
    """
    if stride <= 1:
        return list(events)
    ordered = sort_events(events)
    return [ev for idx, ev in enumerate(ordered) if (idx % stride) == 0]


def _bin_index(t: Timestamp, t0: Timestamp, bin_width: int) -> int:
    return int((int(t) - int(t0)) // int(bin_width))


def chunk_by_bins(
    events: Sequence[Event],
    *,
    window: Window,
    bin_width: int,
) -> Mapping[int, List[Event]]:
    """
    Group events into fixed-width time bins within a window [t0,t1].

    Conventions:
        - Bin index k covers [t0 + k*bin_width, t0 + (k+1)*bin_width - 1]
        - Events outside the window are ignored.

    Returns:
        Dict: bin_index -> list of events in that bin (stable by input order).
    """
    t0, t1 = window
    if t1 < t0 or bin_width <= 0:
        return {}

    out: Dict[int, List[Event]] = {}
    for e in events:
        if not (t0 <= e.t <= t1):
            continue
        k = _bin_index(e.t, t0, bin_width)
        out.setdefault(k, []).append(e)
    return out


def sliding_windows(
    events: Sequence[Event],
    *,
    window_size: int,
    step: int,
    t_start: Optional[Timestamp] = None,
    t_end: Optional[Timestamp] = None,
) -> Iterator[Tuple[Window, List[Event]]]:
    """
    Yield overlapping windows of fixed size covering [t_start, t_end] (or data span).

    Args:
        window_size: size of each window (inclusive span)
        step: shift between consecutive windows
        t_start: optional start; defaults to min timestamp in events
        t_end: optional end; defaults to max timestamp in events

    Yields:
        (window, events_in_window) pairs
    """
    if window_size <= 0 or step <= 0:
        return
    if not events:
        return

    e_sorted = sort_events(events)
    data_t_min = int(e_sorted[0].t)
    data_t_max = int(e_sorted[-1].t)
    t0 = int(t_start) if t_start is not None else data_t_min
    t1 = int(t_end) if t_end is not None else data_t_max
    if t1 < t0:
        return

    w_start = t0
    while w_start <= t1:
        w_end = min(t1, w_start + window_size)
        w = (w_start, w_end)
        in_window = time_window(e_sorted, w)
        yield (w, in_window)
        # Avoid infinite loop if window_size == 0 (already handled)
        w_start += step


__all__ = [
    "time_window",
    "neuron_filter",
    "time_normalize",
    "sort_events",
    "subsample",
    "chunk_by_bins",
    "sliding_windows",
]