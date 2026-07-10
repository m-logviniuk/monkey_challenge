"""Precision/recall thresholding and the BCa bootstrap on known samples."""

import numpy as np
import pytest

from monkey.metrics import (
    bca_ci,
    bootstrap_ci,
    pr_curve,
    precision_recall,
    precision_recall_from_counts,
    reliability,
)


def test_precision_recall_from_counts_strict_threshold():
    tp_probs = np.array([0.95, 0.5])
    fp_probs = np.array([0.6, 0.3])
    # threshold 0.4 (strict >): keep 0.95, 0.5 (TP) and 0.6 (FP); drop 0.3.
    pr = precision_recall_from_counts(tp_probs, fp_probs, total_pos=3,
                                      threshold=0.4)
    assert pr["tp"] == 2 and pr["fp"] == 1 and pr["fn"] == 1
    assert pr["precision"] == pytest.approx(2 / 3)
    assert pr["recall"] == pytest.approx(2 / 3)


def test_precision_recall_high_threshold():
    tp_probs = np.array([0.95, 0.5])
    fp_probs = np.array([0.6, 0.3])
    pr = precision_recall_from_counts(tp_probs, fp_probs, total_pos=3,
                                      threshold=0.9)
    assert pr["tp"] == 1 and pr["fp"] == 0 and pr["fn"] == 2
    assert pr["precision"] == pytest.approx(1.0)
    assert pr["recall"] == pytest.approx(1 / 3)


def test_precision_recall_end_to_end():
    gt = np.array([[0.0, 0.0, 1.0, 0.0], [20.0, 0.0, 1.0, 0.0]])
    pred = np.array([[0.0, 0.0, 0.95, 0.0], [20.0, 0.0, 0.95, 0.0],
                     [99.0, 99.0, 0.2, 0.0]])
    pr = precision_recall(pred, gt, mpp=1.0, prob_threshold=0.4,
                          class_name="lymphocyte")
    assert pr["tp"] == 2 and pr["fp"] == 0 and pr["fn"] == 0


def test_bca_ci_brackets_point_estimate():
    rng = np.random.RandomState(0)
    values = rng.normal(0.5, 0.1, size=40).tolist()
    res = bca_ci(values, lambda s: float(np.mean(s)), n_resamples=500, seed=1)
    assert res["point"] == pytest.approx(float(np.mean(values)))
    assert res["lo"] <= res["point"] <= res["hi"]
    assert res["sd"] > 0.0
    assert res["n"] == 40


def test_bca_ci_reproducible():
    values = [0.2, 0.3, 0.5, 0.4, 0.6, 0.35, 0.45, 0.55]
    a = bca_ci(values, lambda s: float(np.mean(s)), n_resamples=300, seed=7)
    b = bca_ci(values, lambda s: float(np.mean(s)), n_resamples=300, seed=7)
    assert a["lo"] == pytest.approx(b["lo"])
    assert a["hi"] == pytest.approx(b["hi"])


def test_bca_ci_single_record_degenerate():
    res = bca_ci([0.7], lambda s: float(np.mean(s)), n_resamples=100)
    assert res["lo"] == res["hi"] == res["point"] == pytest.approx(0.7)
    assert res["sd"] == 0.0


def test_bootstrap_ci_wrapper():
    values = np.linspace(0.1, 0.9, 30).tolist()
    lo, hi = bootstrap_ci(values, n_resamples=400, seed=3)
    assert lo <= float(np.mean(values)) <= hi


def test_pr_curve_monotone_recall():
    tp_probs = np.array([0.9, 0.7, 0.5])
    fp_probs = np.array([0.6, 0.2])
    curve = pr_curve(tp_probs, fp_probs, total_pos=3,
                     thresholds=np.array([0.0, 0.4, 0.8]))
    # Recall never increases as the score threshold rises.
    assert curve["recall"] == sorted(curve["recall"], reverse=True)
    assert len(curve["precision"]) == 3


def test_reliability_bins_precision():
    # High-score points are all TP, low-score points all FP -> good calibration.
    tp_probs = np.array([0.95, 0.85])
    fp_probs = np.array([0.15, 0.05])
    cal = reliability(tp_probs, fp_probs, n_bins=10)
    assert cal["precision"][-1] == pytest.approx(1.0)  # top bin: all TP
    assert cal["precision"][0] == pytest.approx(0.0)   # bottom bin: all FP
