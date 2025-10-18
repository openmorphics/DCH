# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Torch-free evaluation utilities for offline predictions.

- EvaluationResult: container for metrics, confusion matrix, and class counts.
- evaluate_predictions: compute metrics from y_true and y_pred or y_score.
- aggregate_results: aggregate multiple EvaluationResult objects.
- StreamingEvaluator: accumulate batches (y_pred mode) and finalize metrics.

Dependencies: numpy only (no torch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .metrics import (  # [python.import()](dch_pipeline/evaluation.py:24)
    EPS,
    accuracy as _accuracy,
    confusion_matrix as _confusion_matrix,
    macro_precision_recall_f1,
    micro_precision_recall_f1,
    precision_recall_f1_per_class,
    top_k_accuracy as _top_k_accuracy,
    top_k_accuracies as _top_k_accuracies,
)


@dataclass(frozen=True)
class EvaluationResult:
    """
    Result of evaluating predictions for a multi-class classification task.

    Attributes:
        metrics: Dictionary containing overall metrics and per-class arrays:
            - accuracy: float
            - macro_precision, macro_recall, macro_f1: float
            - micro_precision, micro_recall, micro_f1: float
            - precision_per_class, recall_per_class, f1_per_class: np.ndarray shape (C,)
        confusion: Confusion matrix of shape (C, C), rows=true, cols=pred.
        support_per_class: np.ndarray shape (C,), number of true samples per class.
        total_samples: Total number of samples (int).
        topk: Optional dict mapping "top{k}" to top-k accuracy (floats), when scores provided.
    """
    metrics: Mapping[str, float | np.ndarray]
    confusion: np.ndarray
    support_per_class: np.ndarray
    total_samples: int
    topk: Optional[Mapping[str, float]] = None


# -------------------------
# Internal helpers
# -------------------------


def _to_int1d(x: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D.")
    if a.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.issubdtype(a.dtype, np.integer):
        af = np.asarray(a, dtype=float)
        if not np.isfinite(af).all():
            raise ValueError(f"{name} contains non-finite values.")
        frac, _ = np.modf(af)
        if not np.all(np.abs(frac) <= EPS):
            raise ValueError(f"{name} contains non-integer values.")
        a = af.astype(np.int64)
    else:
        a = a.astype(np.int64)
    return a


def _to_float2d(x: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array of shape (n_samples, n_classes).")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"{name} is empty.")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values.")
    return a


def _validate_classes_param(classes: Optional[Sequence[int]]) -> Optional[np.ndarray]:
    if classes is None:
        return None
    c_arr = _to_int1d(classes, "classes")
    c = int(c_arr.shape[0])
    if c <= 0:
        raise ValueError("classes must be non-empty.")
    # Ensure indices are exactly 0..C-1
    if set(int(x) for x in c_arr.tolist()) != set(range(c)):
        raise ValueError("classes must enumerate 0..C-1 without gaps.")
    return np.arange(c, dtype=np.int64)  # canonicalized


def _infer_num_classes(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray],
    y_score: Optional[np.ndarray],
    classes: Optional[Sequence[int]],
    num_classes: Optional[int],
) -> int:
    c_from_param = None
    if classes is not None:
        c_arr = _validate_classes_param(classes)
        c_from_param = int(c_arr.shape[0])  # type: ignore[union-attr]
    elif num_classes is not None:
        if int(num_classes) <= 0:
            raise ValueError("num_classes must be positive.")
        c_from_param = int(num_classes)

    if y_score is not None:
        c_from_score = int(y_score.shape[1])
        if c_from_param is not None and c_from_param != c_from_score:
            raise ValueError(
                f"num_classes/classes ({c_from_param}) must match y_score.shape[1] ({c_from_score})."
            )
        return c_from_score

    # y_pred mode
    if c_from_param is not None:
        return c_from_param

    if y_pred is None:
        raise ValueError("Internal error: y_pred is None when inferring classes.")
    max_label = int(max(int(y_true.max()), int(y_pred.max())))
    return max_label + 1


def _support_from_cm(cm: np.ndarray) -> np.ndarray:
    return cm.sum(axis=1).astype(np.int64, copy=False)


# -------------------------
# Public API
# -------------------------


def evaluate_predictions(
    y_true: Sequence[int] | np.ndarray,
    *,
    y_pred: Optional[Sequence[int] | np.ndarray] = None,
    y_score: Optional[Sequence[Sequence[float]] | np.ndarray] = None,
    classes: Optional[Sequence[int]] = None,
    num_classes: Optional[int] = None,
    topk: Optional[Sequence[int]] = None,
) -> EvaluationResult:
    """
    Evaluate predictions for multi-class classification.

    Exactly one of y_pred or y_score must be provided.

    Args:
        y_true: Array-like of ints, shape (N,)
        y_pred: Array-like of ints, shape (N,) (optional if y_score provided)
        y_score: Array-like of floats, shape (N, C) (optional if y_pred provided)
        classes: Optional explicit class list enumerating [0..C-1]
        num_classes: Optional explicit number of classes C
        topk: Optional list of k-values for top-k accuracy (only valid with y_score)

    Returns:
        EvaluationResult with metrics, confusion matrix, supports, and optional top-k.

    Raises:
        ValueError: on invalid inputs, empty arrays, or shape mismatches.
    """
    yt = _to_int1d(y_true, "y_true")

    if (y_pred is None and y_score is None) or (y_pred is not None and y_score is not None):
        raise ValueError("Provide exactly one of y_pred or y_score.")

    score_arr: Optional[np.ndarray] = None
    yp: np.ndarray

    if y_score is not None:
        score_arr = _to_float2d(y_score, "y_score")
        if score_arr.shape[0] != yt.shape[0]:
            raise ValueError("y_true and y_score must have the same number of samples.")
        C = _infer_num_classes(yt, None, score_arr, classes, num_classes)
        # Bounds check on y_true against C from scores or params
        if np.any(yt < 0) or np.any(yt >= C):
            raise ValueError(f"y_true contains labels outside the valid range [0, {C-1}].")
        yp = np.argmax(score_arr, axis=1).astype(np.int64)
        # Optional top-k
        topk_metrics: Optional[Dict[str, float]] = None
        if topk is not None:
            topk_metrics = {}
            for k in topk:
                topk_metrics[f"top{int(k)}"] = _top_k_accuracy(yt, score_arr, int(k))
    else:
        yp = _to_int1d(y_pred, "y_pred")  # type: ignore[arg-type]
        if yp.shape[0] != yt.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")
        C = _infer_num_classes(yt, yp, None, classes, num_classes)
        # Bounds check when C provided/inferred
        if np.any(yt < 0) or np.any(yt >= C):
            raise ValueError(f"y_true contains labels outside the valid range [0, {C-1}].")
        if np.any(yp < 0) or np.any(yp >= C):
            raise ValueError(f"y_pred contains labels outside the valid range [0, {C-1}].")
        topk_metrics = None
        if topk is not None:
            # In y_pred mode, top-k metrics cannot be computed without scores
            raise ValueError("topk metrics require y_score; cannot compute from y_pred only.")

    cm = _confusion_matrix(yt, yp, num_classes=C)
    precision_pc, recall_pc, f1_pc = precision_recall_f1_per_class(cm)
    macro_p, macro_r, macro_f1 = macro_precision_recall_f1(precision_pc, recall_pc, f1_pc)
    micro_p, micro_r, micro_f1 = micro_precision_recall_f1(cm)
    acc = _accuracy(yt, yp)
    support = _support_from_cm(cm)
    metrics: Dict[str, float | np.ndarray] = {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "precision_per_class": precision_pc,
        "recall_per_class": recall_pc,
        "f1_per_class": f1_pc,
    }

    return EvaluationResult(
        metrics=metrics,
        confusion=cm,
        support_per_class=support,
        total_samples=int(yt.shape[0]),
        topk=topk_metrics,
    )


def aggregate_results(results: Sequence[EvaluationResult]) -> Dict[str, Any]:
    """
    Aggregate a list of EvaluationResult objects.

    Returns a dictionary with:
        - scalar_means: dict of means for scalar metrics across runs
        - scalar_stds: dict of std (ddof=1) for scalar metrics across runs
        - vector_means: dict of elementwise means for per-class arrays
        - confusion_sum: summed confusion matrix across runs
        - topk_means: dict of means for top-k metrics (only across results that include them)
        - topk_stds: dict of std for top-k metrics
    """
    if len(results) == 0:
        raise ValueError("results must be a non-empty sequence.")

    # Identify scalar and vector keys from the first result
    first = results[0]
    scalar_keys: list[str] = []
    vector_keys: list[str] = []
    for k, v in first.metrics.items():
        if isinstance(v, np.ndarray):
            vector_keys.append(k)
        else:
            scalar_keys.append(k)

    # Scalars
    scalar_means: Dict[str, float] = {}
    scalar_stds: Dict[str, float] = {}
    for k in scalar_keys:
        vals = np.array([float(r.metrics[k]) for r in results], dtype=float)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        scalar_means[k] = mean
        scalar_stds[k] = std

    # Vectors (elementwise mean)
    vector_means: Dict[str, np.ndarray] = {}
    for k in vector_keys:
        arrs = [np.asarray(r.metrics[k], dtype=float) for r in results]
        # Ensure same shape
        shape0 = arrs[0].shape
        if not all(a.shape == shape0 for a in arrs):
            raise ValueError(f"Vector metric '{k}' has inconsistent shapes across results.")
        stacked = np.stack(arrs, axis=0)
        vector_means[k] = stacked.mean(axis=0)

    # Confusion sum
    confusion_sum = np.zeros_like(first.confusion, dtype=np.int64)
    for r in results:
        if r.confusion.shape != confusion_sum.shape:
            raise ValueError("Confusion matrix shapes differ across results.")
        confusion_sum += r.confusion

    # Top-k aggregates (optional)
    # Collect all encountered top-k keys
    topk_keys: set[str] = set()
    for r in results:
        if r.topk:
            topk_keys.update(r.topk.keys())
    topk_means: Dict[str, float] = {}
    topk_stds: Dict[str, float] = {}
    for k in sorted(topk_keys):
        vals = np.array(
            [float(r.topk[k]) for r in results if (r.topk is not None and k in r.topk)],
            dtype=float,
        )
        if vals.size == 0:
            continue
        topk_means[k] = float(vals.mean())
        topk_stds[k] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0

    return {
        "scalar_means": scalar_means,
        "scalar_stds": scalar_stds,
        "vector_means": vector_means,
        "confusion_sum": confusion_sum,
        "topk_means": topk_means,
        "topk_stds": topk_stds,
    }


class StreamingEvaluator:
    """
    Streaming evaluator that accumulates a running confusion matrix for y_pred mode.

    Limitations:
        - Only y_pred is supported in add_batch.
        - Using y_score raises NotImplementedError in this subtask.

    Usage:
        se = StreamingEvaluator(num_classes=3)
        se.add_batch(y_true_b1, y_pred=y_pred_b1)
        se.add_batch(y_true_b2, y_pred=y_pred_b2)
        result = se.finalize()
    """

    def __init__(
        self,
        *,
        classes: Optional[Sequence[int]] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        c_arr = _validate_classes_param(classes)
        if c_arr is not None:
            self._C: Optional[int] = int(c_arr.shape[0])
        elif num_classes is not None:
            if int(num_classes) <= 0:
                raise ValueError("num_classes must be positive.")
            self._C = int(num_classes)
        else:
            self._C = None  # dynamic

        self._cm: Optional[np.ndarray] = None
        self._n: int = 0

    def _ensure_cm(self, needed_C: int) -> None:
        if needed_C <= 0:
            raise ValueError("num_classes must be positive.")
        if self._cm is None:
            self._cm = np.zeros((needed_C, needed_C), dtype=np.int64)
            self._C = needed_C
        else:
            C_curr = int(self._cm.shape[0])
            if needed_C > C_curr:
                new_cm = np.zeros((needed_C, needed_C), dtype=np.int64)
                new_cm[:C_curr, :C_curr] = self._cm
                self._cm = new_cm
                self._C = needed_C

    def add_batch(
        self,
        y_true: Sequence[int] | np.ndarray,
        *,
        y_pred: Optional[Sequence[int] | np.ndarray] = None,
        y_score: Optional[Sequence[Sequence[float]] | np.ndarray] = None,
    ) -> None:
        """
        Add a batch of predictions. Only y_pred mode is supported.

        Raises:
            NotImplementedError: if y_score is provided.
            ValueError: on shape or label validation errors.
        """
        if y_score is not None:
            raise NotImplementedError("StreamingEvaluator supports only y_pred in this subtask.")

        yt = _to_int1d(y_true, "y_true")
        yp = _to_int1d(y_pred, "y_pred") if y_pred is not None else None
        if yp is None:
            raise ValueError("y_pred must be provided for streaming evaluation.")
        if yt.shape[0] != yp.shape[0]:
            raise ValueError("y_true and y_pred must have the same length.")

        # Determine/expand number of classes
        if self._C is not None:
            C_needed = self._C
            # Also validate bounds
            if np.any(yt < 0) or np.any(yt >= C_needed):
                raise ValueError(f"y_true contains labels outside the valid range [0, {C_needed-1}].")
            if np.any(yp < 0) or np.any(yp >= C_needed):
                raise ValueError(f"y_pred contains labels outside the valid range [0, {C_needed-1}].")
        else:
            C_needed = int(max(int(yt.max()), int(yp.max())) + 1)

        self._ensure_cm(C_needed)
        assert self._cm is not None
        cm_batch = _confusion_matrix(yt, yp, num_classes=int(self._C))
        self._cm += cm_batch
        self._n += int(yt.shape[0])

    def finalize(self) -> EvaluationResult:
        """
        Compute EvaluationResult from accumulated statistics.

        Raises:
            ValueError: if no samples have been added.
        """
        if self._cm is None or self._n == 0 or self._C is None:
            raise ValueError("No data added to the StreamingEvaluator.")

        cm = self._cm.copy()
        precision_pc, recall_pc, f1_pc = precision_recall_f1_per_class(cm)
        macro_p, macro_r, macro_f1 = macro_precision_recall_f1(precision_pc, recall_pc, f1_pc)
        micro_p, micro_r, micro_f1 = micro_precision_recall_f1(cm)
        acc = float(np.trace(cm) / max(float(cm.sum()), EPS))
        support = _support_from_cm(cm)
        metrics: Dict[str, float | np.ndarray] = {
            "accuracy": float(acc),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "micro_f1": float(micro_f1),
            "precision_per_class": precision_pc,
            "recall_per_class": recall_pc,
            "f1_per_class": f1_pc,
        }
        return EvaluationResult(
            metrics=metrics,
            confusion=cm,
            support_per_class=support,
            total_samples=int(self._n),
            topk=None,
        )


# -------------------------
# P2 reproducible protocol utilities (opt-in; stdlib-only I/O)
# -------------------------

def config_fingerprint(obj: dict) -> str:
    """
    Compute a stable configuration fingerprint.

    Determinism:
    - Uses JSON dumps with sorted keys and minimal separators to produce a stable string.
    - Hashes with SHA-256 and returns the hex digest.

    Args:
        obj: JSON-serializable mapping (dict).

    Returns:
        Hex string of the SHA-256 digest.

    Notes:
        - Only stdlib is used. If an object is not JSON-serializable, a best-effort
          fallback converts unknown types via str() to maintain determinism for tests.
    """
    import hashlib
    import json

    try:
        s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except TypeError:
        # Fallback: convert unknown objects to strings deterministically
        s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def summarize_reliability(
    hg: "HypergraphOps",
    alpha0: float = 1.0,
    beta0: float = 1.0,
    ci_level: float = 0.95,
    max_edges: int = 10,
) -> dict:
    """
    Summarize hyperedge reliability with Beta posterior mean and MC credible intervals.

    Determinism:
    - CIs are computed via Monte Carlo using a fixed seed (0) and a small sample size (n=2000)
      to keep runtime fast and results reproducible for smoke tests.

    Method:
    - Take a snapshot of the hypergraph and iterate over up to max_edges edges.
      Selection heuristic: sort by last_update_t (desc), then by reliability (desc).
    - For each edge, compute:
        alpha_post, beta_post = posterior_params(edge, alpha0, beta0)
        mean = posterior_mean(alpha_post, beta_post)
        ci = credible_interval_mc(alpha_post, beta_post, level=ci_level, n=2000, seed=0)
    - Return:
        {
          "mean_reliability": float,  # mean of per-edge posterior means over the selected set (0 if no edges)
          "edges": [{"id": str, "mean": float, "ci": [lo, hi]}, ...]
        }

    Args:
        hg: HypergraphOps implementation (e.g., InMemoryHypergraph).
        alpha0: Beta prior alpha.
        beta0: Beta prior beta.
        ci_level: Credible interval mass (e.g., 0.95).
        max_edges: Limit on number of edges to include.

    Returns:
        Dict with aggregate mean and per-edge summaries.
    """
    from dch_core.beta_utils import posterior_params, posterior_mean, credible_interval_mc

    snap = hg.snapshot()
    edges = list(snap.hyperedges.values())
    if not edges:
        return {"mean_reliability": 0.0, "edges": []}

    def _sort_key(e):
        lut = e.last_update_t if e.last_update_t is not None else -1
        return (int(lut), float(e.reliability))

    edges_sorted = sorted(edges, key=_sort_key, reverse=True)
    selected = edges_sorted[: max(0, int(max_edges))]

    items = []
    means = []
    for e in selected:
        a_post, b_post = posterior_params(e, alpha0=alpha0, beta0=beta0)
        m = float(posterior_mean(a_post, b_post))
        lo, hi = credible_interval_mc(a_post, b_post, level=ci_level, n=2000, seed=0)
        items.append({"id": str(e.id), "mean": m, "ci": [float(lo), float(hi)]})
        means.append(m)

    agg_mean = float(sum(means) / len(means)) if means else 0.0
    return {"mean_reliability": agg_mean, "edges": items}


def run_quick_synthetic(spec: dict, artifacts_path: str) -> dict:
    """
    Execute a minimal, deterministic synthetic streaming run and emit a metrics JSONL artifact.

    Scope and intent:
    - Smoke-level reproducibility utility, not a full experiments runner.
    - Uses stdlib-only I/O, hashing, and timestamps.
    - Keeps runtime << 2s by constructing a tiny in-memory pipeline and a 2-event sequence.

    Steps:
      1) Seeds and environment:
         - Build SeedConfig from spec["seeds"] and call set_global_seeds().
         - Capture env via get_environment_fingerprint().
      2) Pipeline:
         - Build PipelineConfig from spec["pipeline_overrides"] with recommended plasticity.impl="beta"
           and DHG delay window overrides.
         - Construct pipeline via DCHPipeline.from_defaults() with a tiny connectivity_map.
      3) Events:
         - Generate a deterministic micro sequence:
           one presyn spike within [delay_min, delay_max] before one head spike.
         - Construct target vertex via make_vertex_id(head_neuron, t_head).
      4) Step:
         - Call pipeline.step(events, target_vertices=[target_vid], sign=+1).
      5) Reliability:
         - Compute summary via summarize_reliability(pipeline.hypergraph, ...).
      6) Artifact:
         - Compute config fingerprint via config_fingerprint(spec).
         - Write a single JSONL record to artifacts_path/metrics.jsonl with fields:
           {
             "ts": ISO8601Z,
             "config_fingerprint": "...",
             "env": {...},
             "metrics": {...},
             "reliability_summary": {...}
           }
      7) Return:
         {"metrics": metrics, "reliability_summary": summary, "artifact": metrics_path}

    Raises:
        Any I/O errors encountered during write are propagated to fail the test.

    Determinism:
    - Seeds applied to Python/NumPy/torch (when available) via set_global_seeds.
    - CI sampling in summarize_reliability uses a fixed seed.
    """
    import os
    import json
    from datetime import datetime
    from dataclasses import asdict

    from dch_core.interfaces import Event, SeedConfig, make_vertex_id
    from dch_pipeline.replay import set_global_seeds, get_environment_fingerprint
    from dch_pipeline.pipeline import DCHPipeline, PipelineConfig, DHGConfig, PlasticityConfig

    # 1) Seeds + environment
    seeds_map = dict(spec.get("seeds", {}) or {})
    seeds = SeedConfig(
        python=int(seeds_map.get("python", 0)),
        numpy=int(seeds_map.get("numpy", 0)),
        torch=int(seeds_map.get("torch", 0)),
        extra={},
    )
    _applied = set_global_seeds(seeds)
    env = get_environment_fingerprint()

    # 2) Pipeline config with overrides (minimal, non-invasive)
    pov = dict(spec.get("pipeline_overrides", {}) or {})
    # DHG delay overrides (optional)
    dhg_ov = dict(pov.get("dhg", {}) or {})
    dhg_cfg = DHGConfig(
        delay_min=int(dhg_ov.get("delay_min", DHGConfig.delay_min)),
        delay_max=int(dhg_ov.get("delay_max", DHGConfig.delay_max)),
    )
    # Plasticity implementation knob (default "ema", recommended "beta")
    plast_ov = dict(pov.get("plasticity", {}) or {})
    plast_impl = str(plast_ov.get("impl", "ema"))
    plast_cfg = PlasticityConfig(impl=plast_impl)

    cfg = PipelineConfig(dhg=dhg_cfg, plasticity=plast_cfg)

    # Connectivity map conversion (keys may be strings in spec)
    conn_spec = dict(spec.get("connectivity", {}) or {})
    conn_map: dict[int, list[int]] = {}
    for k, v in conn_spec.items():
        try:
            hk = int(k)
        except Exception:
            # Skip non-integerable keys to keep utility robust
            continue
        tails = []
        for vi in (v or []):
            try:
                tails.append(int(vi))
            except Exception:
                continue
        conn_map[hk] = tails

    pipeline, _enc = DCHPipeline.from_defaults(cfg=cfg, connectivity_map=conn_map)

    # 3) Deterministic micro event sequence
    if not conn_map:
        # Degenerate case: build a trivial one-edge connectivity to keep smoke path alive
        conn_map = {2: [1]}
    head_neuron = sorted(conn_map.keys())[0]
    tail_list = sorted(conn_map.get(head_neuron, []) or [head_neuron])  # fallback self if empty
    tail_neuron = tail_list[0]

    t_head = 10_000
    dmin = int(cfg.dhg.delay_min)
    dmax = int(cfg.dhg.delay_max)
    # Midpoint delay ensures [dmin, dmax] inclusion
    mid_delay = int((dmin + dmax) // 2)
    if mid_delay < dmin:
        mid_delay = dmin
    if mid_delay > dmax:
        mid_delay = dmax
    t_presyn = int(t_head - mid_delay)

    events = [
        Event(neuron_id=tail_neuron, t=t_presyn, meta=None),
        Event(neuron_id=head_neuron, t=t_head, meta=None),
    ]
    target_vid = make_vertex_id(head_neuron, t_head)

    # 4) Single step with supervision target and positive reinforcement
    metrics = dict(
        pipeline.step(events, target_vertices=[target_vid], sign=+1)
    )

    # 5) Reliability summary (Beta prior defaults 1.0/1.0)
    summary = summarize_reliability(pipeline.hypergraph, alpha0=1.0, beta0=1.0, ci_level=0.95, max_edges=10)

    # 6) Persist metrics JSONL artifact (stdlib-only I/O)
    metrics_dir = str(artifacts_path)
    metrics_path = os.path.join(metrics_dir, "metrics.jsonl")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "config_fingerprint": config_fingerprint(spec),
        "env": asdict(env),
        "metrics": metrics,
        "reliability_summary": summary,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

    # 7) Return compact result
    return {"metrics": metrics, "reliability_summary": summary, "artifact": metrics_path}


def run_quick_dataset(spec: dict, artifacts_dir: str) -> dict:
    """
    Run a minimal, deterministic streaming experiment for a dataset mode and emit a JSONL artifact.

    This helper is opt-in and designed to be non-invasive. It reuses the existing pipeline
    orchestration to process a very small number of synthetic streaming steps, labeled for a chosen
    dataset ("nmnist" or "dvs_gesture"). It avoids heavy downloads and external dependencies,
    keeping CI/runtime fast and reproducible.

    Spec (dict) fields:
      - dataset: one of {"nmnist","dvs_gesture"}
      - config_path: optional path to a YAML config. If None, defaults to:
          - configs/experiments/nmnist.yaml for dataset="nmnist"
          - configs/experiments/dvs_gesture.yaml for dataset="dvs_gesture"
      - seeds: {"python": int, "numpy": int, "torch": int}
      - pipeline_overrides: dict; minimally support {"plasticity": {"impl": "beta"}}
        The overrides are shallow-applied into the pipeline config used here (without mutating the
        loaded YAML mapping). Unknown keys are ignored.
      - limit: optional small integer capping number of streaming steps (default small, if absent)

    Steps:
      - Ensure artifacts_dir exists; set seeds via set_global_seeds().
      - Load the config from config_path if provided (best-effort; will run even if YAML is absent).
      - Apply/merge pipeline_overrides into the loaded config (no mutation of original).
      - Run the DCHPipeline in streaming mode for a tiny number of steps, respecting "limit".
        For config_path == configs/micro.yaml, fall back to the synthetic runner path to maximize reuse.
      - Collect deterministic metrics (sum of step counters) and reliability_summary.
      - Compute cfg_fp via config_fingerprint() using {"dataset","config_path","overrides"}.
      - Write a single-line JSONL at artifacts_dir/metrics.jsonl containing at least:
          {"dataset","config_path","config_fingerprint","metrics","reliability_summary"}.
      - Return {"metrics","reliability_summary","artifact"}.

    Determinism:
    - Seeds are forwarded to set_global_seeds().
    - Reliability CIs use fixed seeds in summarize_reliability().
    - No wall-clock timings are recorded in metrics to avoid platform variance.
    """
    import os
    import json

    from dch_core.interfaces import Event, SeedConfig, make_vertex_id
    from dch_pipeline.replay import set_global_seeds
    from dch_pipeline.pipeline import DCHPipeline, PipelineConfig, DHGConfig, PlasticityConfig

    # 1) Validate dataset and resolve config path defaults
    dataset = str(spec.get("dataset", "")).lower()
    if dataset not in {"nmnist", "dvs_gesture"}:
        raise ValueError("spec['dataset'] must be one of {'nmnist','dvs_gesture'}")

    config_path = spec.get("config_path")
    if not config_path:
        config_path = (
            "configs/experiments/nmnist.yaml"
            if dataset == "nmnist"
            else "configs/experiments/dvs_gesture.yaml"
        )
    config_path_str = str(config_path)

    # 2) Seeds
    seeds_map = dict(spec.get("seeds", {}) or {})
    seeds = SeedConfig(
        python=int(seeds_map.get("python", 0)),
        numpy=int(seeds_map.get("numpy", 0)),
        torch=int(seeds_map.get("torch", 0)),
        extra={},
    )
    _ = set_global_seeds(seeds)

    # 3) Load YAML if available (best-effort; stdlib-only fallback when PyYAML absent)
    loaded_cfg_map: dict = {}
    try:
        import importlib

        yaml = importlib.import_module("yaml")
        try:
            with open(config_path_str, "r", encoding="utf-8") as f:
                data = getattr(yaml, "safe_load")(f) or {}
            if isinstance(data, dict):
                loaded_cfg_map = data
        except Exception:
            loaded_cfg_map = {}
    except Exception:
        # PyYAML not present or other import issue — proceed without consuming the YAML contents
        loaded_cfg_map = {}

    # 4) Merge pipeline_overrides into loaded config (do not mutate originals)
    pov = dict(spec.get("pipeline_overrides", {}) or {})

    def _deep_update(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in (b or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _deep_update(out[k], v)
            else:
                out[k] = v
        return out

    merged_cfg_map = _deep_update(loaded_cfg_map, pov)

    # 5) If micro.yaml is requested, reuse the synthetic runner path but emit dataset-labeled artifact
    if os.path.normpath(config_path_str).endswith(os.path.normpath("configs/micro.yaml")):
        # Reuse the synthetic helper to produce deterministic metrics, then write our dataset artifact.
        # Use a temp subdir to avoid clobbering the final artifact path.
        tmp_dir = os.path.join(str(artifacts_dir), "_synthetic_tmp")
        spec_synth = {
            "seeds": seeds_map,
            "pipeline_overrides": pov,
            "connectivity": {"2": [1]},
        }
        synth_res = run_quick_synthetic(spec_synth, tmp_dir)
        metrics = dict(synth_res.get("metrics", {}))
        reliability_summary = dict(synth_res.get("reliability_summary", {}))
    else:
        # 6) Build a tiny pipeline (reuse orchestration; avoid dataset I/O)
        dhg_ov = dict(pov.get("dhg", {}) or {})
        plast_ov = dict(pov.get("plasticity", {}) or {})
        plast_impl = str(plast_ov.get("impl", "ema"))

        dhg_cfg = DHGConfig(
            delay_min=int(dhg_ov.get("delay_min", DHGConfig.delay_min)),
            delay_max=int(dhg_ov.get("delay_max", DHGConfig.delay_max)),
        )
        plast_cfg = PlasticityConfig(impl=plast_impl)
        cfg = PipelineConfig(dhg=dhg_cfg, plasticity=plast_cfg)

        # Small, static connectivity to keep candidate generation alive
        connectivity = {"2": [1]}
        pipeline, _enc = DCHPipeline.from_defaults(cfg=cfg, connectivity_map=connectivity)

        # 7) Stream a tiny sequence, respecting 'limit' (emulated early stop)
        limit = int(spec.get("limit", 50)) if spec.get("limit", None) is not None else 50
        limit = max(1, min(int(limit), 100))  # defensively bound work

        # Deterministic event generation based on delay window
        head_neuron = 2
        tail_neuron = 1
        dmin = int(cfg.dhg.delay_min)
        dmax = int(cfg.dhg.delay_max)
        mid_delay = int((dmin + dmax) // 2)
        if mid_delay < dmin:
            mid_delay = dmin
        if mid_delay > dmax:
            mid_delay = dmax

        totals: dict = {}
        base_head_t = 10_000
        for i in range(limit):
            t_head = base_head_t + i * (dmax + 10)
            t_tail = int(t_head - mid_delay)

            events = [
                Event(neuron_id=tail_neuron, t=t_tail, meta=None),
                Event(neuron_id=head_neuron, t=t_head, meta=None),
            ]
            target_vid = make_vertex_id(head_neuron, t_head)
            m = pipeline.step(events, target_vertices=[target_vid], sign=+1)
            # Sum numeric counters deterministically
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    totals[k] = totals.get(k, 0) + (int(v) if isinstance(v, bool) else v)

        metrics = totals
        # 8) Reliability summary on final hypergraph snapshot
        reliability_summary = summarize_reliability(pipeline.hypergraph, alpha0=1.0, beta0=1.0, ci_level=0.95, max_edges=10)

    # 9) Persist artifact (required keys only to keep stable across platforms)
    os.makedirs(str(artifacts_dir), exist_ok=True)
    artifact_path = os.path.join(str(artifacts_dir), "metrics.jsonl")
    record = {
        "dataset": dataset,
        "config_path": config_path_str,
        "config_fingerprint": config_fingerprint(
            {"dataset": dataset, "config_path": config_path_str, "overrides": pov}
        ),
        "metrics": metrics,
        "reliability_summary": reliability_summary,
    }
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

    return {"metrics": metrics, "reliability_summary": reliability_summary, "artifact": artifact_path}


__all__ = [
    "EvaluationResult",
    "evaluate_predictions",
    "aggregate_results",
    "StreamingEvaluator",
    "config_fingerprint",
    "summarize_reliability",
    "run_quick_synthetic",
    "run_quick_dataset",
    "reliability_ci_time_series",
    "effect_size_time_series",
    "aggregate_runs",
]
# -------------------------
# Statistical validation utilities (P2-5)
# -------------------------

def _z_for_alpha(alpha: float) -> float:
    """
    Approximate two-sided z critical value for given alpha.
    Supports common alphas without external dependencies.
    """
    a = float(alpha)
    # Common levels
    if abs(a - 0.10) <= 1e-9:
        return 1.6448536269514722
    if abs(a - 0.05) <= 1e-9:
        return 1.959963984540054
    if abs(a - 0.01) <= 1e-9:
        return 2.5758293035489004
    # Fallback to 0.05 if unknown
    return 1.959963984540054


def _record_label(rec: Mapping[str, Any], idx: int) -> str:
    """
    Derive a stable label for alignment:
    - Prefer config_fingerprint when present
    - Else prefer '__name' (optionally injected by loaders/CLI)
    - Else fallback to synthetic index 't{idx}'
    """
    cf = rec.get("config_fingerprint")
    if isinstance(cf, str) and cf:
        return cf
    nm = rec.get("__name") or rec.get("name")
    if isinstance(nm, str) and nm:
        return nm
    return f"t{int(idx)}"


def _aggregate_ci_from_edges(rs: Mapping[str, Any], default_mean: float) -> tuple[float, float]:
    """
    Aggregate an overall CI from per-edge credible intervals by simple averaging.
    If edges are missing, fall back to [mean, mean].
    """
    edges = rs.get("edges") or []
    los: list[float] = []
    his: list[float] = []
    for e in edges:
        try:
            ci = e.get("ci")
            if isinstance(ci, (list, tuple)) and len(ci) == 2:
                lo, hi = float(ci[0]), float(ci[1])
                los.append(lo)
                his.append(hi)
        except Exception:
            continue
    if los and his and len(los) == len(his):
        lo_avg = float(sum(los) / len(los))
        hi_avg = float(sum(his) / len(his))
        # Ensure bounds
        lo_avg = max(0.0, min(1.0, lo_avg))
        hi_avg = max(0.0, min(1.0, hi_avg))
        if lo_avg <= hi_avg:
            return lo_avg, hi_avg
    # Fallback: a degenerate CI around mean
    m = float(default_mean)
    m = max(0.0, min(1.0, m))
    return m, m


def reliability_ci_time_series(
    records: Sequence[Mapping[str, Any]],
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Build a time series of reliability means and aggregated CIs from artifact records.

    Each record is expected to contain a 'reliability_summary' with:
      - 'mean_reliability': float
      - 'edges': list of {'mean': float, 'ci': [lo, hi], ...}

    Returns:
      {
        'alpha': float,
        'names': list[str],  # labels for each point (config_fingerprint or fallback)
        'mean': list[float], # mean reliability per record
        'ci': list[[lo, hi]],# aggregated CI per record (averaged across edges)
      }
    """
    names: list[str] = []
    means: list[float] = []
    cis: list[list[float]] = []

    for i, rec in enumerate(records):
        rs = rec.get("reliability_summary") or {}
        m = float(rs.get("mean_reliability", 0.0))
        lo, hi = _aggregate_ci_from_edges(rs, default_mean=m)
        names.append(_record_label(rec, i))
        means.append(float(m))
        cis.append([float(lo), float(hi)])

    return {
        "alpha": float(alpha),
        "names": names,
        "mean": means,
        "ci": cis,
    }


def _sigma_from_ci(ci_lo: float, ci_hi: float, z: float) -> float:
    """
    Infer an approximate standard deviation from a two-sided CI via:
      sigma ~= (hi - lo) / (2 * z)
    """
    width = float(ci_hi) - float(ci_lo)
    if z <= 0.0:
        return 0.0
    sigma = float(width) / (2.0 * float(z))
    return max(0.0, float(sigma))


def _n_eff_from_mean_and_se(mean: float, se: float) -> float:
    """
    Estimate an effective sample size for a Bernoulli mean using:
      se = sqrt(p*(1-p)/n)  =>  n ~= p*(1-p) / se^2
    Guard against edge cases by clamping p into (0,1) and handling se==0.
    """
    p = max(1e-12, min(1.0 - 1e-12, float(mean)))
    if se <= 0.0:
        return float("inf")
    return float(p * (1.0 - p) / (se * se))


def effect_size_time_series(
    records_a: Sequence[Mapping[str, Any]],
    records_b: Sequence[Mapping[str, Any]],
    *,
    align_by: str = "index",  # "index" | "name"
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compute per-timepoint effect sizes (Cohen's d and Hedges' g) between two series of
    artifact records, aligned either by index or by a stable name.

    Alignment:
    - align_by="index": pairs records by position, up to the min length.
    - align_by="name": pairs records by label derived from config_fingerprint (preferred)
      falling back to '__name' (set by CLI loader) or a synthetic index if unavailable.

    Sigma inference:
    - Uses CI-to-sigma approximation with z = z_{1 - alpha/2}:
        sigma ~= (ci_hi - ci_lo) / (2 * z)
    - Pooled sigma for d: s = sqrt((sigma_a^2 + sigma_b^2) / 2)

    Hedges' g:
    - Applies small-sample correction J(df) ~= 1 - 3/(4*df - 1) with
      df ~= n_a + n_b - 2 where n is estimated from mean and SE.

    Returns:
      {
        'alpha': float,
        'align_by': 'index' | 'name',
        'pairs': list[[name_a, name_b]],
        'cohen_d': list[float],
        'hedges_g': list[float],
      }
    """
    import math

    z = _z_for_alpha(alpha)
    pairs: list[tuple[str, str, Mapping[str, Any], Mapping[str, Any]]] = []

    if align_by == "name":
        map_a: dict[str, tuple[int, Mapping[str, Any]]] = { _record_label(r, i): (i, r) for i, r in enumerate(records_a) }
        map_b: dict[str, tuple[int, Mapping[str, Any]]] = { _record_label(r, i): (i, r) for i, r in enumerate(records_b) }
        # Intersect by name; sort by name for determinism
        names = sorted(set(map_a.keys()) & set(map_b.keys()))
        for nm in names:
            _, ra = map_a[nm]
            _, rb = map_b[nm]
            pairs.append((nm, nm, ra, rb))
    else:
        n = min(len(records_a), len(records_b))
        for i in range(n):
            ra = records_a[i]
            rb = records_b[i]
            na = _record_label(ra, i)
            nb = _record_label(rb, i)
            pairs.append((na, nb, ra, rb))

    cohen_d: list[float] = []
    hedges_g: list[float] = []
    out_pairs: list[list[str]] = []

    for idx, (na, nb, ra, rb) in enumerate(pairs):
        rsa = ra.get("reliability_summary") or {}
        rsb = rb.get("reliability_summary") or {}

        ma = float(rsa.get("mean_reliability", 0.0))
        mb = float(rsb.get("mean_reliability", 0.0))

        loa, hia = _aggregate_ci_from_edges(rsa, default_mean=ma)
        lob, hib = _aggregate_ci_from_edges(rsb, default_mean=mb)

        # Infer sigma from CI; guard against zeros
        sig_a = _sigma_from_ci(loa, hia, z)
        sig_b = _sigma_from_ci(lob, hib, z)
        s_pool = math.sqrt(max(0.0, (sig_a * sig_a + sig_b * sig_b) / 2.0))
        if s_pool <= 0.0:
            d = 0.0
        else:
            d = (ma - mb) / s_pool

        # Hedges' g correction using n_eff estimates
        se_a = sig_a  # assuming CI corresponds to ~1 sd for approximation of SE
        se_b = sig_b
        n_a = _n_eff_from_mean_and_se(ma, se_a)
        n_b = _n_eff_from_mean_and_se(mb, se_b)
        df = max(1.0, (n_a + n_b - 2.0))
        if df > 3.0:
            J = 1.0 - 3.0 / (4.0 * df - 1.0)
        else:
            J = 1.0
        g = float(J) * float(d)

        out_pairs.append([str(na), str(nb)])
        cohen_d.append(float(d))
        hedges_g.append(float(g))

    return {
        "alpha": float(alpha),
        "align_by": str(align_by),
        "pairs": out_pairs,
        "cohen_d": cohen_d,
        "hedges_g": hedges_g,
    }


def aggregate_runs(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Aggregate reliability summaries across groups of artifact records.

    Inputs:
      - groups: dict name -> list of artifact records (JSON objects emitted by quick runners)
      - alpha: CI-to-sigma parameter, forwarded to reliability_ci_time_series for consistency

    Behavior:
      For each group:
        - Compute reliability_ts via reliability_ci_time_series on the group’s records
        - Compute group_mean = average of mean_reliability across records
        - Return per-group summary: {"reliability_ts": ..., "group_mean": float, "n": int}

    Returns:
      {
        "alpha": float,
        "groups": {
          group_name: {"reliability_ts": {...}, "group_mean": float, "n": int}
        }
      }
    """
    out_groups: Dict[str, Any] = {}
    for name, recs in (groups or {}).items():
        rec_list = list(recs or [])
        series = reliability_ci_time_series(rec_list, alpha=alpha)

        means: list[float] = []
        for rec in rec_list:
            rs = rec.get("reliability_summary") or {}
            m = float(rs.get("mean_reliability", 0.0))
            means.append(m)
        group_mean = float(sum(means) / len(means)) if len(means) > 0 else 0.0

        out_groups[str(name)] = {
            "reliability_ts": series,
            "group_mean": float(group_mean),
            "n": int(len(rec_list)),
        }

    return {
        "alpha": float(alpha),
        "groups": out_groups,
    }