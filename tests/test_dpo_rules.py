# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Tests for the minimal DPO engine (GROW, PRUNE, FREEZE) and optional pipeline wiring.

Deterministic, offline, quick:
- Uses in-memory hypergraph and small synthetic events
- No datasets, no subprocesses
"""

from __future__ import annotations

from typing import List

import pytest

from dch_core.interfaces import Event, make_vertex_id, VertexId
from dch_core.hypergraph_mem import InMemoryHypergraph
from dch_core.dpo import DPOEngine, DPOGraphAdapter, DPO_Rule, DPO_LKR, DPO_Match
from dch_pipeline.pipeline import DCHPipeline, PipelineConfig, DPOConfig


def _ingest_two_vertices(hg: InMemoryHypergraph, tail_neuron: int = 1, head_neuron: int = 2, dt: int = 500):
    """
    Ingest two events: tail at t0, head at t0+dt.
    Returns (tail_vid, head_vid, t0, t1)
    """
    t0 = 1000
    t1 = t0 + int(dt)
    v_tail = hg.ingest_event(Event(neuron_id=tail_neuron, t=t0))
    v_head = hg.ingest_event(Event(neuron_id=head_neuron, t=t1))
    return v_tail.id, v_head.id, t0, t1


def _make_unary_edge(hg: InMemoryHypergraph, tail_vid: VertexId, head_vid: VertexId, t_head: int, *, reliability: float = 0.10, s: int = 0, f: int = 0):
    from dch_core.interfaces import Hyperedge, make_edge_id
    tail_set = {tail_vid}
    e = Hyperedge(
        id=make_edge_id(head=head_vid, tail=tail_set, t=t_head),
        tail=tail_set,
        head=head_vid,
        delta_min=0,
        delta_max=int(max(0, t_head - int(str(tail_vid).split("@")[1]))),
        refractory_rho=0,
        reliability=float(reliability),
        counts_success=int(s),
        counts_miss=int(f),
        provenance="unit-test",
    )
    admitted = hg.insert_hyperedges([e])
    assert admitted == [e.id]
    return e.id


def test_dpo_grow_adds_edge():
    # Create empty graph with two vertices and adapter/engine
    hg = InMemoryHypergraph()
    tail_vid, head_vid, t0, t1 = _ingest_two_vertices(hg, tail_neuron=1, head_neuron=2, dt=500)
    adapter = DPOGraphAdapter(hg)
    eng = DPOEngine()

    # Build and apply a GROW rule
    rule = DPO_Rule(
        name="grow_1_to_2",
        kind="GROW",
        lkr=DPO_LKR(),
        preconditions={},
        params={"tails": [tail_vid], "head": head_vid},
    )
    res = eng.apply(rule, DPO_Match(vertices=[tail_vid, head_vid]), adapter)
    assert res.applied, f"GROW should apply, got reason={res.reason}"

    # Validate exactly one edge exists from tail -> head with frozen=False
    snap = hg.snapshot()
    assert len(snap.hyperedges) == 1
    e = next(iter(snap.hyperedges.values()))
    assert e.head == head_vid
    assert e.tail == {tail_vid}
    assert e.attributes.get("frozen", False) is False


def test_dpo_prune_removes_low_reliability():
    hg = InMemoryHypergraph()
    tail_vid, head_vid, t0, t1 = _ingest_two_vertices(hg, tail_neuron=3, head_neuron=4, dt=400)
    # Posterior mean with Beta(1,1): (1+s)/(2+s+f); choose low mean by setting s low, f high
    eid = _make_unary_edge(hg, tail_vid, head_vid, t1, reliability=0.9, s=0, f=10)
    adapter = DPOGraphAdapter(hg)
    eng = DPOEngine(theta_prune=0.5)

    rule = DPO_Rule(
        name="prune_low",
        kind="PRUNE",
        lkr=DPO_LKR(),
        preconditions={"theta_prune": 0.5},
        params={"edge_id": eid, "theta_prune": 0.5},
    )
    res = eng.apply(rule, DPO_Match(edge_id=eid), adapter)
    assert res.applied, f"PRUNE should apply, got reason={res.reason}"
    assert hg.get_edge(eid) is None, "Edge should be removed after PRUNE"


def test_dpo_freeze_marks_high_reliability():
    hg = InMemoryHypergraph()
    tail_vid, head_vid, t0, t1 = _ingest_two_vertices(hg, tail_neuron=5, head_neuron=6, dt=300)
    # High posterior mean: s high, f low
    eid = _make_unary_edge(hg, tail_vid, head_vid, t1, reliability=0.1, s=12, f=0)
    adapter = DPOGraphAdapter(hg)
    eng = DPOEngine(theta_freeze=0.9)

    rule = DPO_Rule(
        name="freeze_high",
        kind="FREEZE",
        lkr=DPO_LKR(),
        preconditions={"theta_freeze": 0.9},
        params={"edge_id": eid, "theta_freeze": 0.9},
    )
    res = eng.apply(rule, DPO_Match(edge_id=eid), adapter)
    assert res.applied, f"FREEZE should apply, got reason={res.reason}"
    e_after = hg.get_edge(eid)
    assert e_after is not None
    assert e_after.attributes.get("frozen", False) is True


def test_pipeline_dpo_flag_off_unchanged():
    # Two pipelines with identical configs; one with dpo=None, one with dpo disabled
    cfg0 = PipelineConfig()
    cfg1 = PipelineConfig(dpo=DPOConfig(enable=False))

    # Connectivity map to ensure a deterministic candidate is generated (1 -> 10)
    connectivity_map = {10: [1]}

    pipe0, _enc0 = DCHPipeline.from_defaults(cfg=cfg0, connectivity_map=connectivity_map)
    pipe1, _enc1 = DCHPipeline.from_defaults(cfg=cfg1, connectivity_map=connectivity_map)

    # Two events: presyn at t=1000, postsyn at t=1500 (Δ=500 within [100,500])
    events = [Event(neuron_id=1, t=1000), Event(neuron_id=10, t=1500)]

    m0 = pipe0.step(events)
    m1 = pipe1.step(events)

    assert m0 == m1, "Metrics should be identical when DPO is disabled (dpo=None vs enable=False)"


def test_pipeline_dpo_grow_enabled():
    # DPO enabled for GROW only
    cfg = PipelineConfig(dpo=DPOConfig(enable=True, apply_ops=("grow",)))
    connectivity_map = {20: [2]}

    pipeline, _enc = DCHPipeline.from_defaults(cfg=cfg, connectivity_map=connectivity_map)

    # Two events: presyn 2@1000, postsyn 20@1500
    events = [Event(neuron_id=2, t=1000), Event(neuron_id=20, t=1500)]
    m = pipeline.step(events)

    # Assert an edge was added and metrics are sane
    snap = pipeline.hypergraph.snapshot()
    assert len(snap.hyperedges) >= 1, "Expected at least one edge from DPO GROW"
    assert int(m.get("n_admitted", 0)) >= 1

    # Validate the edge connectivity matches [tail]->head
    e = next(iter(snap.hyperedges.values()))
    head_vid = make_vertex_id(20, 1500)
    tail_vid = make_vertex_id(2, 1000)
    assert e.head == head_vid
    assert tail_vid in e.tail