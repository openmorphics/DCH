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


__all__ = [
    "EvaluationResult",
    "evaluate_predictions",
    "aggregate_results",
    "StreamingEvaluator",
]