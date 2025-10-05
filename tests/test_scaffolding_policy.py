# -*- coding: utf-8 -*-
import pytest

from dch_core.scaffolding import (
    ScaffoldingParams,
    DefaultScaffoldingPolicy,
    ISOLATE,
    REUSE,
)


def test_similarity_and_decision_isolate():
    params = ScaffoldingParams()
    policy = DefaultScaffoldingPolicy(params)
    policy.register_task("A", {"e1", "e2", "e3"}, {"e1": 0.9, "e2": 0.9, "e3": 0.9})

    action, base = policy.decide({"e4", "e5"})
    assert action == ISOLATE
    assert base is None


def test_similarity_and_decision_reuse():
    params = ScaffoldingParams(similarity_threshold=0.5)
    policy = DefaultScaffoldingPolicy(params)
    policy.register_task("A", {"e1", "e2", "e3"}, {"e1": 0.9, "e2": 0.9, "e3": 0.9})

    action, base = policy.decide({"e2", "e3", "eX"})
    assert action == REUSE
    assert base == "A"


def test_plan_freeze_top_k_and_cutoff():
    params = ScaffoldingParams(
        freeze_strategy="resistance",
        freeze_strength=0.8,
        freeze_top_k=0.5,
        reliability_cutoff=0.6,
    )
    policy = DefaultScaffoldingPolicy(params)

    sig = {"e1", "e2", "e3", "e4"}
    rel = {"e1": 0.95, "e2": 0.9, "e3": 0.7, "e4": 0.5}
    policy.register_task("A", sig, rel)

    mask = policy.plan_freeze("A")
    assert set(mask.keys()) == {"e1", "e2"}
    assert mask["e1"] == pytest.approx(0.8)
    assert mask["e2"] == pytest.approx(0.8)


def test_apply_update_resistance_resistance_mode():
    params = ScaffoldingParams(freeze_strategy="resistance")
    policy = DefaultScaffoldingPolicy(params)

    freeze_mask = {"eF": 0.75}

    # Frozen edge scaled by (1 - resistance)
    assert policy.apply_update_resistance("eF", 1.0, freeze_mask) == pytest.approx(0.25)

    # Unfrozen edge unchanged
    assert policy.apply_update_resistance("eU", 1.0, freeze_mask) == pytest.approx(1.0)


def test_apply_update_resistance_immutable_mode():
    params = ScaffoldingParams(freeze_strategy="immutable")
    policy = DefaultScaffoldingPolicy(params)

    freeze_mask = {"eF": 1.0}

    # Frozen edge becomes 0.0
    assert policy.apply_update_resistance("eF", 1.0, freeze_mask) == pytest.approx(0.0)

    # Unfrozen edge unaffected
    assert policy.apply_update_resistance("eU", 1.0, freeze_mask) == pytest.approx(1.0)


def test_region_tag_deterministic():
    params = ScaffoldingParams(region_prefix="task")
    policy = DefaultScaffoldingPolicy(params)
    assert policy.region_tag("T1") == "task:T1"


def test_decide_picks_closest_task():
    params = ScaffoldingParams(similarity_threshold=0.5)
    policy = DefaultScaffoldingPolicy(params)

    policy.register_task("A", {"e1", "e2", "e3"}, {"e1": 0.9, "e2": 0.9, "e3": 0.9})
    policy.register_task(
        "B",
        {"e2", "e3", "e4", "e5"},
        {"e2": 0.9, "e3": 0.9, "e4": 0.9, "e5": 0.9},
    )

    action, base = policy.decide({"e2", "e3", "e4"})
    assert action == REUSE
    assert base == "B"