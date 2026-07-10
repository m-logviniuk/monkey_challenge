"""FROC matching and scoring on synthetic points with known outcomes.

The matching and FROC-curve logic mirror the official MONKEY evaluation
(``match_coordinates`` and ``monai.metrics`` FROC); these tests pin the exact
true-positive / false-positive counts and the sensitivity curve they produce.
"""

import numpy as np
import pytest

from monkey.froc import (
    case_match,
    froc_curve_data,
    froc_from_match,
    froc_score,
    froc_score_at,
    match_coordinates,
    match_points,
)


def test_match_coordinates_tp_fp_counts():
    gt = np.array([[0.0, 0.0], [10.0, 0.0]])
    pred = np.array([[0.4, 0.0], [9.0, 0.0], [100.0, 100.0]])
    prob = np.array([0.9, 0.8, 0.7])
    out = match_coordinates(gt, pred, prob, radius_px=2.0)
    assert out["n_tp"] == 2
    assert out["n_fp"] == 1
    assert out["n_fn"] == 0
    assert sorted(out["tp_probs"].tolist()) == [0.8, 0.9]
    assert out["fp_probs"].tolist() == [0.7]


def test_match_respects_radius():
    gt = np.array([[0.0, 0.0]])
    pred = np.array([[5.0, 0.0]])
    out = match_coordinates(gt, pred, np.array([0.5]), radius_px=2.0)
    assert out["n_tp"] == 0 and out["n_fn"] == 1 and out["n_fp"] == 1


def test_prediction_claimed_once():
    # Two GT dots near a single prediction: only one can be a true positive.
    gt = np.array([[0.0, 0.0], [1.0, 0.0]])
    pred = np.array([[0.5, 0.0]])
    out = match_coordinates(gt, pred, np.array([0.9]), radius_px=2.0)
    assert out["n_tp"] == 1 and out["n_fn"] == 1 and out["n_fp"] == 0


def test_match_points_masks_agree():
    gt = np.array([[0.0, 0.0], [10.0, 0.0]])
    pred = np.array([[0.4, 0.0], [100.0, 0.0]])
    tp_mask, gt_mask = match_points(pred, gt, radius_px=2.0)
    assert tp_mask.tolist() == [True, False]
    assert gt_mask.tolist() == [True, False]


def test_froc_curve_data_hand_calc():
    tp_probs = np.array([0.9, 0.8])
    fp_probs = np.array([0.7])
    fp_per_mm2, sens = froc_curve_data(fp_probs, tp_probs, total_pos=2,
                                       area_mm2=1.0)
    # Thresholds sweep {0.8, 0.9} then the appended zero point.
    assert np.allclose(fp_per_mm2, [0.0, 0.0, 0.0])
    assert np.allclose(sens, [1.0, 0.5, 0.0])


def test_froc_score_at_matches_interp():
    fp_per_mm2 = np.array([8.0, 4.0, 2.0, 0.0])
    sens = np.array([0.9, 0.7, 0.5, 0.0])
    froc, per_point = froc_score_at(fp_per_mm2, sens, [1, 2, 4])
    expected = np.interp([1, 2, 4], [0.0, 2.0, 4.0, 8.0],
                         [0.0, 0.5, 0.7, 0.9])
    assert np.allclose(list(per_point.values()), expected)
    assert froc == pytest.approx(float(np.mean(expected)))


def test_froc_all_matched_no_fp():
    # Both GT dots are matched with zero false positives. MONAI sweeps the
    # curve over all_probs[1:] (dropping the lowest score), so the captured
    # sensitivity tops out at 0.5 for two points -> this is the official value.
    gt = np.array([[0.0, 0.0, 1.0, 0.0], [20.0, 0.0, 1.0, 0.0]])
    pred = np.array([[0.0, 0.0, 0.9, 0.0], [20.0, 0.0, 0.8, 0.0]])
    res = froc_score(pred, gt, area_mm2=1.0, mpp=1.0, class_name="lymphocyte")
    assert res["froc"] == pytest.approx(0.5)
    assert res["n_tp"] == 2 and res["n_fp"] == 0


def test_froc_no_predictions_scores_zero():
    gt = np.array([[0.0, 0.0, 1.0, 0.0]])
    pred = np.zeros((0, 4))
    res = froc_score(pred, gt, area_mm2=1.0, mpp=1.0, class_name="lymphocyte")
    assert res["froc"] == 0.0
    assert all(v == 0.0 for v in res["sensitivities"].values())


def test_class_filtering_in_case_match():
    # One lymphocyte (class 0) and one monocyte (class 1) GT and pred.
    gt = np.array([[0.0, 0.0, 1.0, 0.0], [50.0, 0.0, 1.0, 1.0]])
    pred = np.array([[0.0, 0.0, 0.9, 0.0], [50.0, 0.0, 0.8, 1.0]])
    lymph = case_match(pred, gt, 1.0, 1.0, class_name="lymphocyte")
    mono = case_match(pred, gt, 1.0, 1.0, class_name="monocyte")
    combined = case_match(pred, gt, 1.0, 1.0, class_name="inflammatory")
    assert lymph["total_pos"] == 1 and lymph["n_pred"] == 1
    assert mono["total_pos"] == 1 and mono["n_pred"] == 1
    assert combined["total_pos"] == 2 and combined["n_pred"] == 2


def test_single_distinct_score_fallback():
    # One TP, no FP -> official "single sensitivity value" fallback (sens=1).
    match = {"tp_probs": np.array([0.9]), "fp_probs": np.zeros(0),
             "total_pos": 1, "area_mm2": 1.0}
    res = froc_from_match(match)
    assert res["froc"] == pytest.approx(1.0)


def test_matches_monai_reference():
    """Cross-check the FROC curve against MONAI when it is installed."""
    monai_metrics = pytest.importorskip("monai.metrics")
    rng = np.random.RandomState(0)
    tp_probs = rng.uniform(0.2, 1.0, size=15)
    fp_probs = rng.uniform(0.0, 0.8, size=25)
    total_pos, area = 20, 3.0
    fp_mine, sens_mine = froc_curve_data(fp_probs, tp_probs, total_pos, area)
    fp_ref, sens_ref = monai_metrics.compute_froc_curve_data(
        fp_probs, tp_probs, total_pos, area)
    assert np.allclose(fp_mine, fp_ref)
    assert np.allclose(sens_mine, sens_ref)
    froc_mine, _ = froc_score_at(fp_mine, sens_mine, [10, 20, 50, 100, 200, 300])
    froc_ref = monai_metrics.compute_froc_score(
        fp_ref, sens_ref, eval_thresholds=(10, 20, 50, 100, 200, 300))
    assert froc_mine == pytest.approx(float(froc_ref))
