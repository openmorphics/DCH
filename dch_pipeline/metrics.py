# Copyright (c) 2025 DCH Maintainers
# License: MIT
"""
Torch-free classification metrics implemented with NumPy only.

Provided functionality:
- Accuracy and top-k accuracy
- Confusion matrix (0..C-1 indexing) with dynamic C discovery
- Precision, Recall, F1 per class from a confusion matrix
- Macro (uniform or weighted) and micro aggregations
- Cosine similarity between vectors
- Small EPS for numerical stability

All functions are pure and side-effect free. Inputs are validated with clear errors.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

# Public numerical stability constant
EPS: float = 1e-12


# -------------------------
# Helpers
# -------------------------


def _as_int_array(x: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    """
    Convert x to a 1-D int64 NumPy array with validation.

    - Ensures non-empty 1-D input.
    - Ensures values are finite integers (no NaN/Inf/float-non-integers).
    - Rejects boolean arrays.
    """
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D.")
    if a.size == 0:
        raise ValueError(f"{name} is empty.")
    if a.dtype == np.bool_:
        raise ValueError(f"{name} must contain integer class labels, not booleans.")
    if not np.issubdtype(a.dtype, np.integer):
        # Validate float-like integers only
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


def _as_float2d_array(x: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    """
    Convert x to a 2-D float64 NumPy array with validation.

    - Ensures non-empty 2-D input with shape (N, C).
    - Ensures values are finite.
    """
    a = np.asarray(x, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array of shape (n_samples, n_classes).")
    if a.size == 0 or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"{name} is empty.")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values.")
    return a


def _check_label_bounds(a: np.ndarray, c: int, name: str) -> None:
    """Validate that all labels are in [0, c-1]."""
    if np.any(a < 0) or np.any(a >= c):
        raise ValueError(f"{name} contains labels outside the valid range [0, {c-1}].")


def _validate_topk(k: int, c: int) -> None:
    if not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer.")
    if k < 1:
        raise ValueError("k must be >= 1.")
    if k > c:
        raise ValueError(f"k={k} cannot exceed number of classes ({c}).")


# -------------------------
# Accuracy and Top-k
# -------------------------


def accuracy(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray) -> float:
    """
    Compute standard accuracy for multi-class predictions.

    Args:
        y_true: Array-like of int labels, shape (N,)
        y_pred: Array-like of int predicted labels, shape (N,)

    Returns:
        Accuracy in [0, 1].

    Raises:
        ValueError: on shape/length mismatch or invalid inputs.
    """
    yt = _as_int_array(y_true, "y_true")
    yp = _as_int_array(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    return float(np.mean(yt == yp))


def top_k_accuracy(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[Sequence[float]] | np.ndarray,
    k: int = 1,
) -> float:
    """
    Compute top-k accuracy from class scores.

    Args:
        y_true: Array-like of int labels, shape (N,)
        y_score: 2-D array-like of floats, shape (N, C) of class scores/probabilities
        k: Integer in [1, C]

    Returns:
        Top-k accuracy in [0, 1].

    Raises:
        ValueError: on invalid shapes, empty input, or k out of bounds.
    """
    yt = _as_int_array(y_true, "y_true")
    score = _as_float2d_array(y_score, "y_score")
    n, c = score.shape
    if yt.shape[0] != n:
        raise ValueError("y_true and y_score must have the same number of samples.")
    _validate_topk(int(k), int(c))
    # top-k indices (unsorted) via argpartition
    # For ties, any consistent selection is acceptable.
    idx_topk = np.argpartition(score, kth=c - k, axis=1)[:, -k:]
    hits = (idx_topk == yt[:, None]).any(axis=1)
    return float(np.mean(hits))


def top_k_accuracies(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[Sequence[float]] | np.ndarray,
    ks: Sequence[int],
) -> Dict[str, float]:
    """
    Compute multiple top-k accuracies.

    Args:
        y_true: int labels, shape (N,)
        y_score: float scores, shape (N, C)
        ks: Iterable of k values

    Returns:
        Dict mapping "top{k}" to accuracy.

    Raises:
        ValueError: if any k is invalid or shapes mismatch.
    """
    score = _as_float2d_array(y_score, "y_score")
    _, c = score.shape
    yt = _as_int_array(y_true, "y_true")
    if yt.shape[0] != score.shape[0]:
        raise ValueError("y_true and y_score must have the same number of samples.")
    out: Dict[str, float] = {}
    for k in ks:
        _validate_topk(int(k), int(c))
        out[f"top{int(k)}"] = top_k_accuracy(yt, score, int(k))
    return out


# -------------------------
# Confusion Matrix
# -------------------------


def confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Compute confusion matrix M of shape (C, C), where:
        rows = true classes, columns = predicted classes.

    Class indices are required to be in [0..C-1]. If num_classes is None,
    C is discovered as max(max(y_true), max(y_pred)) + 1.

    Args:
        y_true: int labels, shape (N,)
        y_pred: int predicted labels, shape (N,)
        num_classes: optional explicit number of classes C

    Returns:
        Confusion matrix array of dtype np.int64 with non-negative counts.

    Raises:
        ValueError: on invalid inputs or labels out of range.
    """
    yt = _as_int_array(y_true, "y_true")
    yp = _as_int_array(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if yt.size == 0:
        raise ValueError("Empty input is not allowed.")
    if num_classes is None:
        c = int(max(int(yt.max()), int(yp.max())) + 1)
    else:
        c = int(num_classes)
    if c <= 0:
        raise ValueError("num_classes must be positive.")
    _check_label_bounds(yt, c, "y_true")
    _check_label_bounds(yp, c, "y_pred")
    # Vectorized bincount construction
    idx = yt * c + yp
    counts = np.bincount(idx, minlength=c * c)
    cm = counts.reshape(c, c).astype(np.int64, copy=False)
    # Post-condition: non-negative counts
    if np.any(cm < 0):
        raise ValueError("Confusion matrix has negative counts, which is invalid.")
    return cm


# -------------------------
# Precision / Recall / F1
# -------------------------


def precision_recall_f1_per_class(cm: np.ndarray, eps: float = EPS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-class precision, recall, and F1 from a confusion matrix.

    Args:
        cm: square confusion matrix with shape (C, C), rows=true, cols=pred
        eps: numerical stability constant

    Returns:
        (precision, recall, f1) arrays of shape (C,)

    Raises:
        ValueError: if cm is not a non-empty square matrix or has negatives.
    """
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.shape[0] == 0:
        raise ValueError("cm must be a non-empty square matrix.")
    if np.any(cm < 0):
        raise ValueError("cm must have non-negative counts.")
    cmf = cm.astype(float, copy=False)
    tp = np.diag(cmf)
    fp = cmf.sum(axis=0) - tp
    fn = cmf.sum(axis=1) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def _weighted_mean(values: np.ndarray, weights: Optional[Sequence[float]], eps: float = EPS) -> float:
    v = np.asarray(values, dtype=float)
    if weights is None:
        return float(v.mean()) if v.size > 0 else 0.0
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.shape[0] != v.shape[0]:
        raise ValueError("weights must be a 1-D array matching the number of classes.")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    denom = float(w.sum())
    if denom <= 0:
        # Uniform fallback if all-zero weights provided
        return float(v.mean()) if v.size > 0 else 0.0
    return float(np.dot(v, w) / (denom + eps))


def macro_precision_recall_f1(
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    weights: Optional[Sequence[float]] = None,
    eps: float = EPS,
) -> Tuple[float, float, float]:
    """
    Macro-averaged precision, recall, and F1.

    If weights is provided, computes a weighted macro average as:
        sum(w_i * metric_i) / sum(w_i).
    If weights is None, uses uniform averaging.

    Args:
        precision, recall, f1: per-class arrays (C,)
        weights: optional non-negative weights of shape (C,)
        eps: numerical stability constant

    Returns:
        (macro_precision, macro_recall, macro_f1)
    """
    mp = _weighted_mean(precision, weights, eps)
    mr = _weighted_mean(recall, weights, eps)
    mf1 = _weighted_mean(f1, weights, eps)
    return mp, mr, mf1


def micro_precision_recall_f1(cm: np.ndarray, eps: float = EPS) -> Tuple[float, float, float]:
    """
    Micro-averaged precision, recall, and F1 from confusion matrix.

    For single-label multi-class classification, micro-precision == micro-recall == accuracy.

    Args:
        cm: confusion matrix (C, C)
        eps: numerical stability constant

    Returns:
        (micro_precision, micro_recall, micro_f1)
    """
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.shape[0] == 0:
        raise ValueError("cm must be a non-empty square matrix.")
    if np.any(cm < 0):
        raise ValueError("cm must have non-negative counts.")
    cmf = cm.astype(float, copy=False)
    tp_total = float(np.trace(cmf))
    total = float(cmf.sum())
    fp_total = float(cmf.sum(axis=0).sum() - np.trace(cmf))  # equals total - tp_total
    fn_total = float(cmf.sum(axis=1).sum() - np.trace(cmf))  # equals total - tp_total
    micro_p = tp_total / (tp_total + fp_total + eps)
    micro_r = tp_total / (tp_total + fn_total + eps)
    micro_f1 = 2.0 * micro_p * micro_r / (micro_p + micro_r + eps)
    return micro_p, micro_r, micro_f1


# -------------------------
# Cosine similarity
# -------------------------


def cosine_similarity(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray, eps: float = EPS) -> float:
    """
    Compute cosine similarity between two vectors using NumPy only.

    cos_sim(a, b) = (a · b) / (||a|| * ||b|| + eps)

    If either vector has zero norm, returns 0.0.

    Args:
        a, b: 1-D vectors with the same length
        eps: small constant to stabilize division

    Returns:
        Cosine similarity in [-1, 1] (approximately due to eps).
    """
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    if va.ndim != 1 or vb.ndim != 1:
        raise ValueError("a and b must be 1-D vectors.")
    if va.size == 0 or vb.size == 0:
        raise ValueError("a and b must be non-empty.")
    if va.shape[0] != vb.shape[0]:
        raise ValueError("a and b must have the same length.")
    if not (np.isfinite(va).all() and np.isfinite(vb).all()):
        raise ValueError("a and b must contain finite values.")
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= EPS or nb <= EPS:
        return 0.0
    sim = float(np.dot(va, vb) / (na * nb + eps))
    # Clamp numerically to [-1, 1]
    return float(np.clip(sim, -1.0, 1.0))


__all__ = [
    "EPS",
    "accuracy",
    "top_k_accuracy",
    "top_k_accuracies",
    "confusion_matrix",
    "precision_recall_f1_per_class",
    "macro_precision_recall_f1",
    "micro_precision_recall_f1",
    "cosine_similarity",
]