# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Statistical testing and reporting utilities for DCH evaluation.

Implements:
- Paired t-test with 95% CI on mean differences
- Wilcoxon signed-rank test with r-effect size approximation
- Cohen's d (paired)
- Cliff's delta (non-parametric effect size)
- Benjamini–Hochberg FDR correction
- Mean and CI helpers (t-based and bootstrap percentile)

These functions are CPU-friendly and rely on NumPy/SciPy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


# -------------------------
# Data structures
# -------------------------


@dataclass(frozen=True)
class TestResult:
    """Generic container for hypothesis test outcomes."""
    statistic: float
    pvalue: float
    extra: Mapping[str, float]  # additional fields (df, ci_low, ci_high, effect sizes, etc.)


# -------------------------
# Helpers
# -------------------------


def _to_array(x: Sequence[float] | np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError("Input must be 1-D.")
    if a.size == 0:
        raise ValueError("Empty input.")
    return a


def _paired_diff(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> np.ndarray:
    a = _to_array(x)
    b = _to_array(y)
    if a.shape != b.shape:
        raise ValueError("For paired tests, x and y must have the same shape.")
    return a - b


# -------------------------
# Mean and Confidence Intervals
# -------------------------


def mean_ci_t(
    x: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    t-based 1-alpha confidence interval for the mean of a sample.
    Returns (mean, ci_low, ci_high).
    """
    a = _to_array(x)
    n = a.size
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
        hw = tcrit * sd / np.sqrt(n)
        return mean, mean - hw, mean + hw
    return mean, mean, mean


def bootstrap_ci(
    x: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap percentile CI for the mean.
    Returns (mean, ci_low, ci_high).
    """
    a = _to_array(x)
    mean = float(a.mean())
    if rng is None:
        rng = np.random.default_rng(0)
    boot = np.empty(n_boot, dtype=float)
    n = a.size
    for i in range(n_boot):
        samp = a[rng.integers(0, n, size=n)]
        boot[i] = float(samp.mean())
    lo = float(np.percentile(boot, 100 * (alpha / 2)))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return mean, lo, hi


# -------------------------
# Parametric and Nonparametric Tests
# -------------------------


def paired_ttest(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """
    Paired t-test with two-sided p-value and 1-alpha CI on mean difference (x - y).
    """
    diff = _paired_diff(x, y)
    n = diff.size
    t_stat, pval = stats.ttest_rel(diff, np.zeros_like(diff), alternative="two-sided")
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd_diff > 0:
        tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
        hw = tcrit * sd_diff / np.sqrt(n)
        ci_low, ci_high = mean_diff - hw, mean_diff + hw
    else:
        ci_low = ci_high = mean_diff
    return TestResult(
        statistic=float(t_stat),
        pvalue=float(pval),
        extra={
            "df": float(n - 1),
            "mean_diff": mean_diff,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        },
    )


def wilcoxon_signed_rank(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    zero_method: str = "wilcox",
    alpha: float = 0.05,
) -> TestResult:
    """
    Wilcoxon signed-rank test (two-sided).
    Approximates effect size r = z / sqrt(n_eff), where z inferred from p-value.

    Note: SciPy returns statistic and p-value; z is reconstructed from p-value
    assuming two-sided normal approximation. The sign of z is taken from median(diff).
    """
    diff = _paired_diff(x, y)
    # Exclude zeros if zero_method dictates
    if zero_method == "wilcox":
        mask = diff != 0
        diff_eff = diff[mask]
    else:
        diff_eff = diff

    if diff_eff.size == 0:
        return TestResult(statistic=0.0, pvalue=1.0, extra={"r_effect": 0.0})

    stat, pval = stats.wilcoxon(diff_eff, zero_method=zero_method, alternative="two-sided", correction=False)
    # Approximate z from two-sided p-value
    # z magnitude from inverse survival; sign from median of differences
    if pval > 0:
        z_abs = float(stats.norm.isf(pval / 2))
        sgn = float(np.sign(np.median(diff_eff)))
        z = sgn * z_abs
        r_effect = float(z / np.sqrt(diff_eff.size))
    else:
        z = np.inf * float(np.sign(np.median(diff_eff)))
        r_effect = 1.0
    return TestResult(statistic=float(stat), pvalue=float(pval), extra={"z": float(z), "r_effect": float(r_effect)})


def cohens_d_paired(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> float:
    """
    Cohen's d for paired samples: d = mean(diff) / sd(diff).
    """
    diff = _paired_diff(x, y)
    sd = float(diff.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(diff.mean() / sd)


def cliffs_delta(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> float:
    """
    Cliff's delta (nonparametric effect size) for two samples x and y.
    Range in [-1, 1], positive when x tends to be larger than y.
    Implementation is O(n*m); sufficient for small n (seed-based evaluation).
    """
    a = _to_array(x)
    b = _to_array(y)
    n, m = a.size, b.size
    greater = 0
    less = 0
    for ai in a:
        greater += int(np.sum(ai > b))
        less += int(np.sum(ai < b))
    return float((greater - less) / (n * m))


# -------------------------
# Multiple testing correction
# -------------------------


def benjamini_hochberg(
    pvalues: Sequence[float] | np.ndarray,
    q: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini–Hochberg FDR control.
    Returns:
        rejected: boolean array indicating which hypotheses are rejected
        pvals_adj: adjusted p-values (BH)
    """
    p = _to_array(pvalues)
    m = p.size
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)

    # Compute adjusted p-values
    p_adj_sorted = p_sorted * m / ranks
    p_adj_sorted = np.minimum.accumulate(p_adj_sorted[::-1])[::-1]
    p_adj = np.empty_like(p_adj_sorted)
    p_adj[order] = np.clip(p_adj_sorted, 0, 1)

    # Determine rejections
    thresh_sorted = (ranks / m) * q
    below = p_sorted <= thresh_sorted
    k = np.max(np.where(below)[0]) + 1 if np.any(below) else 0
    rejected_sorted = np.zeros(m, dtype=bool)
    if k > 0:
        rejected_sorted[:k] = True
    rejected = np.empty_like(rejected_sorted)
    rejected[order] = rejected_sorted
    return rejected, p_adj


# -------------------------
# Aggregation helpers
# -------------------------


def aggregate_runs(
    values: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> Mapping[str, float]:
    """
    Compute mean, std (ddof=1), and t-based CI for a sequence of run metrics.
    """
    a = _to_array(values)
    mean, ci_low, ci_high = mean_ci_t(a, alpha=alpha)
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci_low": ci_low, "ci_high": ci_high, "n": float(a.size)}


__all__ = [
    "TestResult",
    "mean_ci_t",
    "bootstrap_ci",
    "paired_ttest",
    "wilcoxon_signed_rank",
    "cohens_d_paired",
    "cliffs_delta",
    "benjamini_hochberg",
    "aggregate_runs",
]