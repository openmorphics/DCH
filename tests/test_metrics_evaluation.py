# Copyright (c) 2025 DCH Maintainers
# License: MIT

import numpy as np
import pytest

from dch_pipeline.metrics import (  # [python.import()](tests/test_metrics_evaluation.py:8)
    accuracy,
    top_k_accuracy,
    top_k_accuracies,
    confusion_matrix,
    precision_recall_f1_per_class,
    macro_precision_recall_f1,
    micro_precision_recall_f1,
    cosine_similarity,
)

from dch_pipeline.evaluation import (  # [python.import()](tests/test_metrics_evaluation.py:20)
    evaluate_predictions,
    aggregate_results,
    StreamingEvaluator,
)


def test_accuracy_and_topk_correctness():
    # Accuracy with label vectors
    y_true = np.array([0, 1, 2, 2], dtype=int)
    y_pred = np.array([0, 2, 2, 1], dtype=int)
    assert accuracy(y_true, y_pred) == pytest.approx(0.5)

    # Top-k with score matrix
    scores = np.array(
        [
            [0.90, 0.05, 0.05],  # true=0 -> top1 correct
            [0.20, 0.60, 0.20],  # true=1 -> top1 correct
            [0.10, 0.30, 0.60],  # true=2 -> top1 correct
            [0.50, 0.40, 0.10],  # true=2 -> top1 wrong, top2 wrong
        ],
        dtype=float,
    )
    y_true2 = np.array([0, 1, 2, 2], dtype=int)

    # top-1 equals accuracy of argmax
    argmax_pred = scores.argmax(axis=1)
    top1 = top_k_accuracy(y_true2, scores, k=1)
    assert top1 == pytest.approx(accuracy(y_true2, argmax_pred))

    # top-2 should be 3/4 = 0.75 (true labels {0,1,2,2}; last row's top2 are {0,1}, not containing 2)
    top2 = top_k_accuracy(y_true2, scores, k=2)
    assert top2 == pytest.approx(0.75)

    # top-3 equals 1.0 for 3 classes
    top3 = top_k_accuracy(y_true2, scores, k=3)
    assert top3 == pytest.approx(1.0)

    # Batch interface for multiple ks
    tk = top_k_accuracies(y_true2, scores, ks=[1, 2, 3])
    assert tk["top1"] == pytest.approx(top1)
    assert tk["top2"] == pytest.approx(top2)
    assert tk["top3"] == pytest.approx(top3)

    # k > C raises
    with pytest.raises(ValueError):
        _ = top_k_accuracy(y_true2, scores, k=4)


def test_confusion_matrix_and_prf1():
    # Simple 3-class example with zero-support class (class 2 has zero true instances)
    y_true = np.array([0, 1, 1, 1], dtype=int)
    y_pred = np.array([1, 1, 2, 2], dtype=int)

    cm = confusion_matrix(y_true, y_pred)  # discover C dynamically
    assert cm.shape == (3, 3)
    expected = np.array(
        [
            [0, 1, 0],  # true 0 -> predicted 1
            [0, 1, 2],  # true 1 -> predicted 1 twice and 2 twice overall (but 3 trues total => 1 tp + 2 fn to class 2)
            [0, 0, 0],  # true 2 -> zero support
        ],
        dtype=np.int64,
    )
    assert np.array_equal(cm, expected)

    # Per-class precision/recall/f1
    prec, rec, f1 = precision_recall_f1_per_class(cm)

    # Hand-computed:
    # class 0: tp=0, col0 sum=0 -> precision=0; row0 sum=1 -> recall=0; f1=0
    # class 1: tp=1, col1 sum=2 -> precision=1/2=0.5; row1 sum=3 -> recall=1/3; f1=0.4
    # class 2: tp=0, col2 sum=2 -> precision=0; row2 sum=0 -> recall=0; f1=0
    assert prec[0] == pytest.approx(0.0)
    assert rec[0] == pytest.approx(0.0)
    assert f1[0] == pytest.approx(0.0)

    assert prec[1] == pytest.approx(0.5)
    assert rec[1] == pytest.approx(1.0 / 3.0)
    assert f1[1] == pytest.approx(0.4)

    assert prec[2] == pytest.approx(0.0)
    assert rec[2] == pytest.approx(0.0)
    assert f1[2] == pytest.approx(0.0)

    # Macro (uniform)
    mp, mr, mf1 = macro_precision_recall_f1(prec, rec, f1)
    assert mp == pytest.approx((0.0 + 0.5 + 0.0) / 3.0)
    assert mr == pytest.approx((0.0 + (1.0 / 3.0) + 0.0) / 3.0)
    assert mf1 == pytest.approx((0.0 + 0.4 + 0.0) / 3.0)

    # Micro equals accuracy for single-label multi-class
    mip, mir, mif1 = micro_precision_recall_f1(cm)
    assert mip == pytest.approx(1.0 / 4.0)
    assert mir == pytest.approx(1.0 / 4.0)
    assert mif1 == pytest.approx(1.0 / 4.0)


def test_evaluation_end_to_end_with_y_pred():
    y_true = np.array([0, 1, 1, 1], dtype=int)
    y_pred = np.array([1, 1, 2, 2], dtype=int)

    res = evaluate_predictions(y_true, y_pred=y_pred)
    assert isinstance(res.metrics, dict)
    assert "accuracy" in res.metrics
    assert "macro_f1" in res.metrics
    assert "micro_f1" in res.metrics
    assert "precision_per_class" in res.metrics
    assert "recall_per_class" in res.metrics
    assert "f1_per_class" in res.metrics

    assert res.confusion.shape == (3, 3)
    assert res.support_per_class.shape == (3,)
    assert res.total_samples == 4

    # Compare with hand-computed values from previous test
    assert res.metrics["accuracy"] == pytest.approx(0.25)
    assert res.metrics["micro_f1"] == pytest.approx(0.25)
    assert res.topk is None  # no scores provided


def test_evaluation_with_y_score_and_topk():
    y_true = np.array([0, 1, 2, 2], dtype=int)
    scores = np.array(
        [
            [0.90, 0.05, 0.05],  # -> pred 0 correct
            [0.20, 0.60, 0.20],  # -> pred 1 correct
            [0.10, 0.30, 0.60],  # -> pred 2 correct
            [0.50, 0.40, 0.10],  # -> pred 0 wrong (true=2)
        ],
        dtype=float,
    )

    res = evaluate_predictions(y_true, y_score=scores, topk=[1, 2, 3])
    assert res.topk is not None
    argmax_pred = scores.argmax(axis=1)

    assert res.topk["top1"] == pytest.approx(accuracy(y_true, argmax_pred))
    assert res.topk["top2"] == pytest.approx(0.75)
    assert res.topk["top3"] == pytest.approx(1.0)
    assert res.metrics["accuracy"] == pytest.approx(res.topk["top1"])

    # k > C raises during evaluation as well
    with pytest.raises(ValueError):
        _ = evaluate_predictions(y_true, y_score=scores, topk=[4])


def test_aggregate_results_means_stds_and_confusion_sum():
    # Two simple runs, same number of classes
    y_true1 = np.array([0, 1, 2, 2], dtype=int)
    y_pred1 = np.array([0, 1, 2, 1], dtype=int)  # acc = 0.75

    y_true2 = np.array([0, 1, 2, 1], dtype=int)
    y_pred2 = np.array([0, 2, 2, 1], dtype=int)  # acc = 0.75

    r1 = evaluate_predictions(y_true1, y_pred=y_pred1)
    r2 = evaluate_predictions(y_true2, y_pred=y_pred2)

    agg = aggregate_results([r1, r2])

    # Scalar means/stds
    acc_vals = np.array([r1.metrics["accuracy"], r2.metrics["accuracy"]], dtype=float)
    assert agg["scalar_means"]["accuracy"] == pytest.approx(float(acc_vals.mean()))
    # identical accuracies -> std == 0
    assert agg["scalar_stds"]["accuracy"] == pytest.approx(0.0)

    # Vector means: check one vector key shape matches and values are averaged
    for key in ("precision_per_class", "recall_per_class", "f1_per_class"):
        vm = agg["vector_means"][key]
        assert vm.shape == r1.metrics[key].shape
        expected = 0.5 * (np.asarray(r1.metrics[key]) + np.asarray(r2.metrics[key]))
        assert np.allclose(vm, expected, atol=1e-12)

    # Confusion sum equals elementwise sum
    assert np.array_equal(agg["confusion_sum"], r1.confusion + r2.confusion)


def test_streaming_evaluator_matches_single_shot():
    # Two batches, 3 classes
    b1_true = np.array([0, 1], dtype=int)
    b1_pred = np.array([0, 2], dtype=int)

    b2_true = np.array([2, 1, 0], dtype=int)
    b2_pred = np.array([2, 1, 1], dtype=int)

    # Streaming
    se = StreamingEvaluator(num_classes=3)
    se.add_batch(b1_true, y_pred=b1_pred)
    se.add_batch(b2_true, y_pred=b2_pred)
    res_stream = se.finalize()

    # Single-shot
    y_true = np.concatenate([b1_true, b2_true], axis=0)
    y_pred = np.concatenate([b1_pred, b2_pred], axis=0)
    res_single = evaluate_predictions(y_true, y_pred=y_pred, num_classes=3)

    assert np.array_equal(res_stream.confusion, res_single.confusion)
    assert res_stream.metrics["accuracy"] == pytest.approx(res_single.metrics["accuracy"])
    assert res_stream.metrics["micro_f1"] == pytest.approx(res_single.metrics["micro_f1"])
    assert np.allclose(
        res_stream.metrics["precision_per_class"],
        res_single.metrics["precision_per_class"],
        atol=1e-12,
    )
    assert np.allclose(
        res_stream.metrics["recall_per_class"],
        res_single.metrics["recall_per_class"],
        atol=1e-12,
    )
    assert np.allclose(res_stream.metrics["f1_per_class"], res_single.metrics["f1_per_class"], atol=1e-12)


def test_streaming_evaluator_scores_not_implemented():
    se = StreamingEvaluator(num_classes=3)
    y_true = np.array([0, 1], dtype=int)
    scores = np.array([[0.8, 0.2, 0.0], [0.1, 0.2, 0.7]], dtype=float)
    with pytest.raises(NotImplementedError):
        se.add_batch(y_true, y_score=scores)


def test_edge_cases_errors():
    # Empty inputs
    with pytest.raises(ValueError):
        _ = evaluate_predictions(np.array([], dtype=int), y_pred=np.array([], dtype=int))

    # Labels out of range when num_classes specified
    with pytest.raises(ValueError):
        _ = evaluate_predictions(np.array([0, 1], dtype=int), y_pred=np.array([0, 3], dtype=int), num_classes=3)

    # top-k greater than number of classes raises
    y_true = np.array([0, 1], dtype=int)
    scores = np.array([[0.7, 0.3, 0.0], [0.2, 0.1, 0.7]], dtype=float)
    with pytest.raises(ValueError):
        _ = evaluate_predictions(y_true, y_score=scores, topk=[4])


def test_cosine_similarity_basic():
    # Identical direction (scaled)
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([2.0, 0.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(1.0)

    # Orthogonal vectors
    a2 = np.array([1.0, 0.0])
    b2 = np.array([0.0, 1.0])
    assert cosine_similarity(a2, b2) == pytest.approx(0.0)