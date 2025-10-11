"""
Benchmark: traversal (torch-free, CPU-only)

Purpose:
- Micro-benchmark the constrained backward traversal on synthetic hypergraphs using the in-memory backend.

Example usage:
- python benchmarks/benchmark_traversal.py --num-neurons 512 --num-edges 10000

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
from typing import List, Sequence, Tuple, Set
from dch_core.interfaces import Event, Hyperedge, Vertex, VertexId, make_edge_id
from dch_core.hypergraph_mem import InMemoryHypergraph
from dch_core.traversal import DefaultTraversalEngine

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
    rank = max(1, int(math.ceil(p * n)))
    return float(xs_sorted[rank - 1])

def _tail_size_from_branching(rng: random.Random, branching_factor: int) -> int:
    """
    Draw an integer tail size near 'branching_factor' with small variance, clamped in [1, 3].
    """
    mu = max(1.0, float(branching_factor))
    val = int(round(rng.gauss(mu, 0.75)))
    if val < 1:
        val = 1
    if val > 3:
        val = 3
    return val

def build_synthetic_hypergraph(
    rng: random.Random,
    num_neurons: int,
    num_edges: int,
    branching_factor: int,
    time_span: float,
) -> Tuple[InMemoryHypergraph, List[VertexId]]:
    """
    Create an in-memory hypergraph with random vertices and hyperedges.
    - Vertices are created implicitly by ingesting Events at chosen times.
    - Each hyperedge's head is later than all tail vertices by a delay in [1, 20].
    Returns: (graph, list_of_head_vertex_ids)
    """
    g = InMemoryHypergraph()
    delay_min = 1
    delay_max = 20
    refractory_rho = 5
    tmax_int = int(max(1.0, time_span))
    heads: List[VertexId] = []
    for _ in range(int(num_edges)):
        head_neu = int(rng.randrange(num_neurons))
        # ensure head time allows room for tail delays
        t_head = int(rng.randint(delay_max + 1, max(delay_max + 1, tmax_int)))
        v_head = g.ingest_event(Event(neuron_id=head_neu, t=t_head))
        tail_size = _tail_size_from_branching(rng, int(branching_factor))
        tail_ids: Set[VertexId] = set()
        attempts = 0
        while len(tail_ids) < tail_size and attempts < tail_size * 6:
            attempts += 1
            tail_neu = int(rng.randrange(num_neurons))
            if tail_neu == head_neu:
                continue
            delay = int(rng.randint(delay_min, delay_max))
            t_tail = t_head - delay
            if t_tail < 0:
                continue
            v_tail = g.ingest_event(Event(neuron_id=tail_neu, t=t_tail))
            tail_ids.add(v_tail.id)
        if not tail_ids:
            continue
        eid = make_edge_id(head=v_head.id, tail=tail_ids, t=t_head)
        e = Hyperedge(
            id=eid,
            tail=set(tail_ids),
            head=v_head.id,
            delta_min=delay_min,
            delta_max=delay_max,
            refractory_rho=refractory_rho,
            reliability=0.10,
            provenance="synthetic",
        )
        g.insert_hyperedges([e])
        heads.append(v_head.id)
    return g, heads

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark_traversal",
        description="Micro-benchmark constrained backward traversal on synthetic hypergraphs (CPU, stdlib-only).",
    )
    parser.add_argument("--num-neurons", type=int, default=512, help="Number of neurons (default: 512)")
    parser.add_argument("--num-edges", type=int, default=5000, help="Number of hyperedges to synthesize (default: 5000)")
    parser.add_argument("--branching-factor", type=int, default=2, help="Average tail size per hyperedge (default: 2)")
    parser.add_argument("--time-span", type=float, default=1000.0, help="Timestamp span [0, time_span) (default: 1000.0)")
    parser.add_argument("--runs", type=int, default=3, help="Number of repetitions to compute median (default: 3)")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed (default: 123)")
    # Modest traversal parameters to keep runtime predictable
    parser.add_argument("--beam", type=int, default=8, help="Beam size for traversal (default: 8)")
    parser.add_argument("--horizon", type=int, default=200, help="Backward time horizon in time units (default: 200)")
    parser.add_argument("--targets", type=int, default=64, help="Number of sink heads to benchmark (default: 64)")
    args = parser.parse_args(argv)

    rng = random.Random(int(args.seed))

    g, head_ids = build_synthetic_hypergraph(
        rng=rng,
        num_neurons=int(args.num_neurons),
        num_edges=int(args.num_edges),
        branching_factor=int(args.branching_factor),
        time_span=float(args.time_span),
    )
    # pick candidate target heads with incoming edges if possible
    unique_heads = list(dict.fromkeys(head_ids))  # preserve insertion order
    if not unique_heads:
        # fallback: use any vertex ids in graph (should not happen with defaults)
        # InMemoryHypergraph doesn't expose a direct vertex iterator here
        pass
    n_targets = max(1, min(int(args.targets), len(unique_heads)))
    # deterministic sample: shuffle with rng seeded above and take first n_targets
    heads_copy = list(unique_heads)
    rng.shuffle(heads_copy)
    target_vids = heads_copy[:n_targets]

    engine = DefaultTraversalEngine()

    durations_ms: List[float] = []
    # Use the same targets across runs for determinism
    for r in range(int(args.runs)):
        for vid in target_vids:
            v = g.get_vertex(vid)
            if v is None:
                continue
            t0 = time.perf_counter()
            _ = engine.backward_traverse(
                hypergraph=g,
                target=v,
                horizon=int(args.horizon),
                beam_size=int(args.beam),
                rng=None,
                refractory_enforce=True,
            )
            t1 = time.perf_counter()
            durations_ms.append((t1 - t0) * 1000.0)

    mean_ms = _mean(durations_ms)
    median_ms = _median(durations_ms)
    p95_ms = _percentile(durations_ms, 0.95)

    result = {
        "benchmark": "traversal",
        "num_neurons": int(args.num_neurons),
        "num_edges": int(args.num_edges),
        "targets": int(n_targets),
        "beam": int(args.beam),
        "horizon": int(args.horizon),
        "runs": int(args.runs),
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "seed": int(args.seed),
    }
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