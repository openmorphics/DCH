"""
Benchmark: pipeline (torch-free, CPU-only)

Purpose:
- Macro-benchmark the DCH pipeline step with synthetic events using in-memory backend.

Example usage:
- python benchmarks/benchmark_pipeline.py --num-neurons 256 --event-rate 200 --steps 500

Notes:
- Zero extra dependencies (stdlib only). Deterministic via --seed.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
import time
from typing import Dict, List, Mapping, Sequence

from dch_core.interfaces import Event
from dch_pipeline.pipeline import DCHPipeline, PipelineConfig

def _mean(xs: Sequence[float]) -> float:
    return (sum(xs) / float(len(xs))) if xs else 0.0

def _median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    mid = n // 2
    if n % 2 == 1:
        return float(xs_sorted[mid])
    return 0.5 * (xs_sorted[mid - 1] + xs_sorted[mid])

def _percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    # rank in [1..n], ceil-style
    rank = max(1, int(math.ceil(p * n)))
    return float(xs_sorted[rank - 1])

def _generate_step_events(
    rng: random.Random,
    num_neurons: int,
    event_rate: int,
    t_start: int,
    dt: int,
) -> Sequence[Event]:
    """
    Generate a list of synthetic events for one step.
    - neuron_id in [0, num_neurons)
    - timestamps strictly increasing: t = t_start + (i+1)*dt
    """
    events: List[Event] = []
    t = t_start
    for _ in range(int(event_rate)):
        t += dt
        nid = int(rng.randrange(num_neurons))
        events.append(Event(neuron_id=nid, t=int(t)))
    return events

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark_pipeline",
        description="Macro-benchmark DCH pipeline step on synthetic events (CPU, stdlib-only).",
    )
    parser.add_argument("--num-neurons", type=int, default=256, help="Number of neurons (default: 256)")
    parser.add_argument("--event-rate", type=int, default=100, help="Events per step (default: 100)")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps (default: 200)")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed (default: 123)")
    parser.add_argument("--enable-abstraction", action="store_true", help="Enable FSM+abstraction path (may be slower)")
    args = parser.parse_args(argv)

    rng = random.Random(int(args.seed))

    # Build pipeline (torch-free path; in-memory backends)
    cfg = PipelineConfig(enable_abstraction=bool(args.enable_abstraction))
    pipeline, _encoder = DCHPipeline.from_defaults(cfg=cfg)

    # Synthetic timeline bounded roughly by steps * 10.0
    # Choose dt so that total span stays within ~steps*10
    steps = int(args.steps)
    event_rate = int(args.event_rate)
    time_span_target = max(1, steps * 10)
    total_events = max(1, steps * event_rate)
    dt = max(1, int(time_span_target // total_events))  # at least 1
    t_cursor = 0

    durations_ms: List[float] = []
    sum_candidates = 0
    sum_admitted = 0
    sum_hyperpaths = 0
    sum_edges_updated = 0
    sum_pruned = 0
    sum_fsm_observed = 0
    sum_fsm_promoted = 0

    for _ in range(steps):
        events = _generate_step_events(rng, int(args.num_neurons), event_rate, t_cursor, dt)
        if events:
            t_cursor = events[-1].t

        t0 = time.perf_counter()
        metrics: Mapping[str, object] = pipeline.step(events)
        t1 = time.perf_counter()

        durations_ms.append((t1 - t0) * 1000.0)

        # Aggregate exposed metrics when available
        sum_candidates += int(metrics.get("n_candidates", 0)) if isinstance(metrics.get("n_candidates", 0), int) else 0
        sum_admitted += int(metrics.get("n_admitted", 0)) if isinstance(metrics.get("n_admitted", 0), int) else 0
        sum_hyperpaths += int(metrics.get("n_hyperpaths", 0)) if isinstance(metrics.get("n_hyperpaths", 0), int) else 0
        sum_edges_updated += int(metrics.get("n_edges_updated", 0)) if isinstance(metrics.get("n_edges_updated", 0), int) else 0
        sum_pruned += int(metrics.get("n_pruned", 0)) if isinstance(metrics.get("n_pruned", 0), int) else 0
        if args.enable_abstraction:
            sum_fsm_observed += int(metrics.get("n_fsm_observed", 0)) if isinstance(metrics.get("n_fsm_observed", 0), int) else 0
            sum_fsm_promoted += int(metrics.get("n_fsm_promoted", 0)) if isinstance(metrics.get("n_fsm_promoted", 0), int) else 0

    result: Dict[str, object] = {
        "benchmark": "pipeline",
        "num_neurons": int(args.num_neurons),
        "event_rate": int(args.event_rate),
        "steps": steps,
        "mean_step_ms": _mean(durations_ms),
        "median_step_ms": _median(durations_ms),
        "p95_step_ms": _percentile(durations_ms, 0.95),
        "total_events": total_events,
        "enable_abstraction": bool(args.enable_abstraction),
        "seed": int(args.seed),
        # Optional aggregates (present for observability; not required by spec)
        "sum_candidates": int(sum_candidates),
        "sum_admitted": int(sum_admitted),
        "sum_hyperpaths": int(sum_hyperpaths),
        "sum_edges_updated": int(sum_edges_updated),
        "sum_pruned": int(sum_pruned),
    }
    if args.enable_abstraction:
        result["sum_fsm_observed"] = int(sum_fsm_observed)
        result["sum_fsm_promoted"] = int(sum_fsm_promoted)

    print(json.dumps(result, sort_keys=True))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        # structured error without traceback spam
        err = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps(err, sort_keys=True))
        sys.exit(1)