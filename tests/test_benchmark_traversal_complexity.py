from __future__ import annotations

from benchmarks.benchmark_traversal_complexity import run_once


def test_traversal_complexity_run_once_small():
    K, L, C_in = 2, 3, 2
    res = run_once(K, L, C_in)

    # Required keys present
    for key in ["benchmark", "K", "L", "C_in_cap", "expansions", "elapsed_ms", "theoretical_ops"]:
        assert key in res, f"missing key: {key}"

    # Types and invariants
    assert res["benchmark"] == "traversal_complexity"
    assert int(res["K"]) == K
    assert int(res["L"]) == L
    assert int(res["C_in_cap"]) == C_in
    assert isinstance(res["expansions"], int) and res["expansions"] > 0
    assert isinstance(res["elapsed_ms"], float) or isinstance(res["elapsed_ms"], int)
    assert int(res["theoretical_ops"]) == K * L * C_in