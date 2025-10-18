# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Micro-benchmark skeleton for Beta–Bernoulli plasticity.

Runs a small synthetic update loop and prints a single JSON line:
{"engine":"beta","edges":N,"paths":M,"updates":U,"mean_reliability":r,"elapsed_ms":t}

Deterministic RNG seeding. Torch is not used.
"""

from __future__ import annotations

import json
import random
import time
from typing import List, Tuple

from dch_core.interfaces import Event, Hyperedge, Hyperpath, PlasticityState, make_vertex_id, make_edge_id
from dch_core.hypergraph_mem import InMemoryHypergraph
from dch_core.plasticity_beta import BetaPlasticityEngine


def _make_edges(hg: InMemoryHypergraph, n: int, head_neuron: int = 10, head_t: int = 10_000) -> List[Hyperedge]:
    """
    Create n unary edges with unique tails, admissible exact windows, and random initial reliability.
    """
    edges: List[Hyperedge] = []
    # Materialize shared head vertex
    hg.ingest_event(Event(neuron_id=head_neuron, t=head_t))
    for i in range(n):
        tail_neuron = 1 + i
        dt = 100 + (i % 25)  # small variety of admissible delays
        tail_t = head_t - dt
        hg.ingest_event(Event(neuron_id=tail_neuron, t=tail_t))
        head_vid = make_vertex_id(head_neuron, head_t)
        tail_vid = make_vertex_id(tail_neuron, tail_t)
        e = Hyperedge(
            id=make_edge_id(head=head_vid, tail={tail_vid}, t=int(head_t + tail_neuron)),
            tail={tail_vid},
            head=head_vid,
            delta_min=dt,
            delta_max=dt,
            refractory_rho=0,
            reliability=random.uniform(0.05, 0.95),
            provenance="benchmark",
        )
        edges.append(e)
    admitted = hg.insert_hyperedges(edges)
    # Keep only admitted order
    eid_to_edge = {e.id: e for e in edges}
    return [eid_to_edge[eid] for eid in admitted if eid in eid_to_edge]


def _make_hyperpaths(edges: List[Hyperedge], m: int) -> List[Hyperpath]:
    """
    Create m hyperpaths; each selects a random edge and assigns a positive score in (0,1].
    """
    hps: List[Hyperpath] = []
    for _ in range(m):
        e = random.choice(edges)
        score = random.random() + 1e-6  # avoid zero
        hps.append(Hyperpath(head=e.head, edges=(e.id,), score=score, length=1, label=str(e.id)))
    return hps


def run_benchmark(n_edges: int = 1000, m_paths: int = 100, n_updates: int = 10) -> Tuple[float, float]:
    random.seed(12345)

    hg = InMemoryHypergraph()
    edges = _make_edges(hg, n_edges)
    hps = _make_hyperpaths(edges, m_paths)

    engine = BetaPlasticityEngine(alpha0=1.0, beta0=1.0)
    pstate = PlasticityState(ema_alpha=0.10, reliability_clamp=(0.02, 0.98), prune_threshold=0.05)

    t0 = time.perf_counter()
    now = 0
    for u in range(n_updates):
        sign = +1 if (u % 2 == 0) else -1
        now += 1
        _ = engine.update_from_evidence(hypergraph=hg, hyperpaths=hps, sign=sign, now_t=now, state=pstate)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    snap = hg.snapshot()
    r_mean = 0.0
    if snap.hyperedges:
        r_mean = sum(float(e.reliability) for e in snap.hyperedges.values()) / float(len(snap.hyperedges))

    return r_mean, elapsed_ms


def main():
    N = 1000
    M = 100
    U = 10
    r_mean, elapsed_ms = run_benchmark(N, M, U)
    print(
        json.dumps(
            {
                "engine": "beta",
                "edges": N,
                "paths": M,
                "updates": U,
                "mean_reliability": r_mean,
                "elapsed_ms": elapsed_ms,
            }
        )
    )


if __name__ == "__main__":
    main()