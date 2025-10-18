# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Traversal complexity micro-benchmark (stdlib-only).

Constructs a small synthetic, layered in-memory hypergraph with:
- C_in: number of incoming unary edges per vertex (branching cap per layer)
- L: maximum "depth" (number of backward expansion layers)
- K: seeds/beam (number of target vertices and beam size for traversal)

Uses:
- Storage: dch_core.hypergraph_mem.InMemoryHypergraph
- Traversal: dch_core.traversal.DefaultTraversalEngine

Emits a single JSON line when run as a script:
{"benchmark":"traversal_complexity","K":K,"L":L,"C_in_cap":C_in,"expansions":E,"elapsed_ms":t,"theoretical_ops":K*L*C_in}

Notes
- Deterministic: no RNG usage; purely synthetic structure.
- Edges are unary (tail size = 1) to isolate the effect of C_in branching.
- Depth is bounded by constructing exactly L+1 time-ordered layers.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List, Tuple

from dch_core.interfaces import Event, Hyperedge, Vertex, VertexId, make_edge_id, make_vertex_id
from dch_core.hypergraph_mem import InMemoryHypergraph
from dch_core.traversal import DefaultTraversalEngine


# Fixed temporal admissibility for synthetic edges
_DELTA_MIN = 100
_DELTA_MAX = 200
_STEP = 150  # satisfies DELTA_MIN <= STEP <= DELTA_MAX


def _build_layered_hypergraph(K: int, L: int, C_in: int) -> Tuple[InMemoryHypergraph, List[VertexId], int]:
    """
    Build a layered hypergraph with (L+1) layers of vertices.
    - Each layer contains N = max(K, C_in) vertices with times t_i = i * STEP.
    - For each head v in layer i (i >= 1), create exactly C_in unary incoming edges
      from distinct vertices in layer i-1.

    Returns:
        (hypergraph, target_vertex_ids, horizon)
    """
    hg = InMemoryHypergraph()
    n_per_layer = max(int(K), int(C_in))
    layers: List[List[Vertex]] = []

    # Create vertices per layer
    for i in range(L + 1):
        t_i = i * _STEP
        layer_vertices: List[Vertex] = []
        for j in range(n_per_layer):
            vid = make_vertex_id(neuron_id=1000 + i * 100 + j, t=t_i)
            v = hg.ingest_event(Event(neuron_id=1000 + i * 100 + j, t=t_i))
            assert v.id == vid
            layer_vertices.append(v)
        layers.append(layer_vertices)

    # Create edges: for each head in layer i (i>=1), add C_in unary edges from layer i-1
    for i in range(1, L + 1):
        prev = layers[i - 1]
        curr = layers[i]
        for head_idx, head_v in enumerate(curr):
            # Pick first C_in tails from previous layer (distinct; stable)
            for tail_idx in range(C_in):
                tail_v = prev[tail_idx % len(prev)]
                e = Hyperedge(
                    id=make_edge_id(head_v.id, [tail_v.id], t=(i * 1000 + head_idx * 10 + tail_idx)),
                    tail={tail_v.id},
                    head=head_v.id,
                    delta_min=_DELTA_MIN,
                    delta_max=_DELTA_MAX,
                    refractory_rho=0,
                    reliability=1.0,  # deterministic, uniform scoring
                )
                hg.insert_hyperedges([e])

    # Targets: choose first K vertices from the last layer
    targets = [v.id for v in layers[-1][:K]]
    # Horizon must include the earliest tail relative to the target
    horizon = L * _STEP + 1
    return hg, targets, horizon


def run_once(K: int, L: int, C_in: int) -> Dict[str, float | int | str]:
    """
    Execute the benchmark once and return the metrics dict.
    """
    K = int(K)
    L = int(L)
    C_in = int(C_in)

    hg, targets, horizon = _build_layered_hypergraph(K=K, L=L, C_in=C_in)
    trav = DefaultTraversalEngine(length_penalty_base=0.98)

    t0 = time.perf_counter()
    expansions = 0
    for vid in targets:
        v = hg.get_vertex(vid)
        if v is None:
            continue
        # Beam size == K (as per spec); rng=None for determinism
        paths = trav.backward_traverse(
            hypergraph=hg,
            target=v,
            horizon=horizon,
            beam_size=K,
            rng=None,
            refractory_enforce=True,
        )
        # Use number of discovered hyperpaths as a proxy for processed states
        expansions += int(len(paths))
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0

    out: Dict[str, float | int | str] = {
        "benchmark": "traversal_complexity",
        "K": K,
        "L": L,
        "C_in_cap": C_in,
        "expansions": int(expansions),
        "elapsed_ms": float(elapsed_ms),
        "theoretical_ops": int(K * L * C_in),
    }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traversal complexity micro-benchmark (stdlib-only)")
    parser.add_argument("--K", type=int, default=4, help="Seeds/beam size (default: 4)")
    parser.add_argument("--L", type=int, default=6, help="Max depth (layers) (default: 6)")
    parser.add_argument("--C_in", type=int, default=8, help="Incoming edges per vertex (default: 8)")
    args = parser.parse_args()

    res = run_once(args.K, args.L, args.C_in)
    # Emit a single JSON line
    print(json.dumps(res, separators=(",", ":")))