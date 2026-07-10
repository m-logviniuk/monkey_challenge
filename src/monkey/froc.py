"""Challenge-exact FROC scoring for inflammatory-cell detection.

This reproduces the behaviour of the official MONKEY evaluation
(``computationalpathologygroup/monkey-challenge`` ``evaluation/evaluate.py``)
so that the numbers reported here are comparable to the leaderboard.

Two pieces are mirrored from the official pipeline:

- **Matching** follows the official ``match_coordinates``: a KD-tree is built
  over the predicted points and each ground-truth dot queries its single
  nearest prediction within the matching radius; a prediction may be claimed
  by only one ground-truth dot. Matched predictions are true positives, their
  scores become ``tp_probs``; unmatched predictions are false positives, their
  scores become ``fp_probs``. This is a nearest-neighbour rule, not optimal
  bipartite matching, and the prediction score order does not affect it -
  both properties match the official script.
- **FROC** reproduces ``monai.metrics.compute_froc_curve_data`` /
  ``compute_froc_score`` (MONAI 0.5.0, as used by the official evaluation):
  the sensitivity / false-positive curve is swept over the union of the
  matched and unmatched scores, false positives are normalised per mm2 by
  treating the ROI area in mm2 as the "image count", and the FROC score is the
  mean sensitivity interpolated at ``FP_POINTS`` = {10, 20, 50, 100, 200, 300}
  false positives per mm2.

Coordinate and unit conventions used across the interface:
- points are ``[N, 4]`` arrays with columns ``(x0, y0, score, class)`` in
  level-0 slide pixels (the space :func:`monkey.detect.detect_case` returns);
- ``mpp`` is microns per level-0 pixel (``base_mpp`` from the HDF5 attrs), used
  to convert the micron matching radius to pixels; matching in level-0 pixels
  with ``radius_um / mpp`` is equivalent to the official matching in mm;
- ``area_mm2`` is the scored ROI area used to normalise false positives per
  square millimetre;
- ``class`` is 0 for lymphocyte and 1 for monocyte; ``class_name`` selects the
  scored subset ("lymphocyte", "monocyte") or all points ("inflammatory").

Deviation from the official script: the official code matches in physical mm
using the fixed level-0 spacing ``0.24199951...`` um/px, while this module
matches in level-0 pixels using each case's own ``base_mpp``. The two are
identical when ``base_mpp`` equals that spacing and differ only by the true
per-case pixel size otherwise, which is the physically correct radius.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .config import CLASS_NAMES, FP_POINTS, MATCH_RADIUS_UM


def _class_index(class_name: str) -> int | None:
    """Class column value for ``class_name``; ``None`` means all classes."""
    if class_name in ("inflammatory", "combined", "all"):
        return None
    if class_name in CLASS_NAMES:
        return CLASS_NAMES.index(class_name)
    raise ValueError(
        f"unknown class_name {class_name!r}; expected one of "
        f"{(*CLASS_NAMES, 'inflammatory')}."
    )


def _select(points: np.ndarray, class_name: str) -> np.ndarray:
    """Return the ``[K, 4]`` rows of ``points`` scored for ``class_name``."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float64)
    ci = _class_index(class_name)
    if ci is None:
        return pts
    return pts[pts[:, 3].astype(int) == ci]


def _radius_px(class_name: str, mpp: float,
               match_radius_um: dict | None) -> float:
    radii = match_radius_um or MATCH_RADIUS_UM
    key = class_name if class_name in radii else "inflammatory"
    return float(radii[key]) / float(mpp)


def match_coordinates(gt_xy: np.ndarray, pred_xy: np.ndarray,
                      pred_prob: np.ndarray, radius_px: float) -> dict:
    """Nearest-neighbour matching mirroring the official ``match_coordinates``.

    ``gt_xy`` and ``pred_xy`` are ``[*, 2]`` coordinate arrays and ``pred_prob``
    the ``[*]`` prediction scores, all in a common (pixel) space; ``radius_px``
    is the matching tolerance in that space. A KD-tree over the predictions is
    queried once per ground-truth dot for its nearest prediction within the
    radius; each prediction can be claimed once. Returns
    ``{"n_tp", "n_fn", "n_fp", "tp_probs", "fp_probs"}``.
    """
    gt_xy = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    pred_xy = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    pred_prob = np.asarray(pred_prob, dtype=np.float64).reshape(-1)
    n_gt, n_pred = len(gt_xy), len(pred_xy)

    if n_gt == 0 and n_pred == 0:
        return {"n_tp": 0, "n_fn": 0, "n_fp": 0,
                "tp_probs": np.zeros(0), "fp_probs": np.zeros(0)}
    if n_pred == 0:
        return {"n_tp": 0, "n_fn": n_gt, "n_fp": 0,
                "tp_probs": np.zeros(0), "fp_probs": np.zeros(0)}
    if n_gt == 0:
        return {"n_tp": 0, "n_fn": 0, "n_fp": n_pred,
                "tp_probs": np.zeros(0), "fp_probs": pred_prob.copy()}

    tree = cKDTree(pred_xy)
    distances, indices = tree.query(gt_xy, distance_upper_bound=radius_px)

    matched_pred: set[int] = set()
    tp_probs: list[float] = []
    for dist, pred_idx in zip(distances, indices, strict=True):
        if pred_idx < n_pred and dist <= radius_px and pred_idx not in matched_pred:
            matched_pred.add(int(pred_idx))
            tp_probs.append(float(pred_prob[pred_idx]))

    n_tp = len(matched_pred)
    fp_probs = [float(pred_prob[i]) for i in range(n_pred) if i not in matched_pred]
    return {
        "n_tp": n_tp,
        "n_fn": n_gt - n_tp,
        "n_fp": n_pred - n_tp,
        "tp_probs": np.asarray(tp_probs, dtype=np.float64),
        "fp_probs": np.asarray(fp_probs, dtype=np.float64),
    }


def match_points(pred_xy: np.ndarray, gt_xy: np.ndarray,
                 radius_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Match predictions to ground truth within ``radius_px``.

    Returns a boolean true-positive mask over the predictions and a boolean
    matched mask over the ground-truth points, using the same nearest-neighbour
    rule as :func:`match_coordinates` (score order does not affect the result).
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    gt_xy = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    n_pred, n_gt = len(pred_xy), len(gt_xy)
    tp_mask = np.zeros(n_pred, dtype=bool)
    gt_mask = np.zeros(n_gt, dtype=bool)
    if n_pred == 0 or n_gt == 0:
        return tp_mask, gt_mask
    tree = cKDTree(pred_xy)
    distances, indices = tree.query(gt_xy, distance_upper_bound=radius_px)
    for g, (dist, pred_idx) in enumerate(zip(distances, indices, strict=True)):
        if pred_idx < n_pred and dist <= radius_px and not tp_mask[pred_idx]:
            tp_mask[pred_idx] = True
            gt_mask[g] = True
    return tp_mask, gt_mask


def case_match(pred_points: np.ndarray, gt_points: np.ndarray, area_mm2: float,
               mpp: float, *, class_name: str = "inflammatory",
               match_radius_um: dict | None = None) -> dict:
    """Match one case for ``class_name`` and return the raw FROC ingredients.

    Returns ``{"tp_probs", "fp_probs", "total_pos", "area_mm2", "n_pred"}``.
    Pooling these across cases and calling :func:`froc_from_match` reproduces
    the official out-of-fold aggregation.
    """
    pred = _select(pred_points, class_name)
    gt = _select(gt_points, class_name)
    radius_px = _radius_px(class_name, mpp, match_radius_um)
    matched = match_coordinates(gt[:, :2], pred[:, :2], pred[:, 2], radius_px)
    return {
        "tp_probs": matched["tp_probs"],
        "fp_probs": matched["fp_probs"],
        "total_pos": int(len(gt)),
        "area_mm2": float(area_mm2),
        "n_pred": int(len(pred)),
    }


def froc_curve_data(fp_probs: np.ndarray, tp_probs: np.ndarray,
                    total_pos: int, area_mm2: float) -> tuple[np.ndarray, np.ndarray]:
    """FROC curve data reproducing ``monai.metrics.compute_froc_curve_data``.

    Returns ``(fp_per_mm2, sensitivity)`` swept over the union of the matched
    and unmatched scores, with false positives normalised by ``area_mm2`` (used
    as the "image count") and sensitivity by ``total_pos``.
    """
    # Vectorised equivalent of MONAI's threshold loop: sweep the union of the
    # distinct scores (dropping the smallest, as MONAI does with all_probs[1:])
    # and count scores >= threshold via searchsorted, then append the zero
    # endpoint. searchsorted keeps this O(n log n) rather than O(unique * n),
    # which matters for the bootstrap over tens of thousands of pooled points.
    fp_probs = np.asarray(fp_probs, dtype=np.float64)
    tp_probs = np.asarray(tp_probs, dtype=np.float64)
    thresholds = np.unique(np.concatenate([fp_probs, tp_probs]))[1:]
    fp_sorted = np.sort(fp_probs)
    tp_sorted = np.sort(tp_probs)
    fp_counts = len(fp_probs) - np.searchsorted(fp_sorted, thresholds, side="left")
    tp_counts = len(tp_probs) - np.searchsorted(tp_sorted, thresholds, side="left")
    total_fps = np.append(fp_counts.astype(np.float64), 0.0)
    total_tps = np.append(tp_counts.astype(np.float64), 0.0)
    fp_per_mm2 = total_fps / float(area_mm2)
    sensitivity = total_tps / float(total_pos)
    return fp_per_mm2, sensitivity


def froc_score_at(fp_per_mm2: np.ndarray, sensitivity: np.ndarray,
                  fp_points: list[int]) -> tuple[float, dict]:
    """Mean interpolated sensitivity at ``fp_points`` (MONAI ``compute_froc_score``).

    Returns ``(froc, {fp: sensitivity})``. Interpolation uses the reversed
    (ascending false-positive) curve exactly as MONAI does, so requesting a
    false-positive rate beyond the observed range clamps to the endpoint.
    """
    xs = np.asarray(fp_per_mm2, dtype=np.float64)[::-1]
    ys = np.asarray(sensitivity, dtype=np.float64)[::-1]
    interp = np.interp(np.asarray(fp_points, dtype=np.float64), xs, ys)
    per_point = {int(p): float(v) for p, v in zip(fp_points, interp, strict=True)}
    return float(np.mean(interp)), per_point


def froc_from_match(match: dict, *, fp_points: list[int] | None = None) -> dict:
    """FROC score + curve from a (pooled) match dict.

    ``match`` carries ``tp_probs``, ``fp_probs``, ``total_pos`` and
    ``area_mm2``. Replicates the official ``get_froc_score`` edge cases: an
    empty curve scores 0, and a single distinct score is handled by the
    "one true positive" fallback the official code uses.
    """
    fp_points = fp_points or FP_POINTS
    tp_probs = np.asarray(match["tp_probs"], dtype=np.float64)
    fp_probs = np.asarray(match["fp_probs"], dtype=np.float64)
    total_pos = int(match["total_pos"])
    area_mm2 = float(match["area_mm2"])
    n_tp = int(len(tp_probs))
    n_fp = int(len(fp_probs))

    empty = {
        "froc": 0.0,
        "sensitivities": {int(p): 0.0 for p in fp_points},
        "fp_per_mm2": [0.0], "sensitivity": [0.0],
        "total_pos": total_pos, "n_tp": n_tp, "n_fp": n_fp,
        "area_mm2": area_mm2,
    }
    if total_pos == 0 or area_mm2 <= 0:
        return empty

    fp_per_mm2, sensitivity = froc_curve_data(fp_probs, tp_probs, total_pos,
                                              area_mm2)
    if len(sensitivity) == 0:
        return empty
    if len(sensitivity) == 1:
        # Official fallback when only one distinct score exists.
        fp_rate = n_fp / area_mm2
        froc = float(np.mean([1.0 if fp_rate < p else 0.0 for p in fp_points]))
        return {
            "froc": froc,
            "sensitivities": {int(p): (1.0 if fp_rate < p else 0.0)
                              for p in fp_points},
            "fp_per_mm2": [float(fp_rate)], "sensitivity": [1.0],
            "total_pos": total_pos, "n_tp": n_tp, "n_fp": n_fp,
            "area_mm2": area_mm2,
        }
    froc, per_point = froc_score_at(fp_per_mm2, sensitivity, fp_points)
    return {
        "froc": froc,
        "sensitivities": per_point,
        "fp_per_mm2": fp_per_mm2.tolist(),
        "sensitivity": sensitivity.tolist(),
        "total_pos": total_pos, "n_tp": n_tp, "n_fp": n_fp,
        "area_mm2": area_mm2,
    }


def _zero_froc(match: dict, fp_points: list[int]) -> dict:
    """Zero-FROC result for a slide with no predictions (official guard)."""
    return {
        "froc": 0.0,
        "sensitivities": {int(p): 0.0 for p in fp_points},
        "fp_per_mm2": [0.0], "sensitivity": [0.0],
        "total_pos": int(match["total_pos"]), "n_tp": 0,
        "n_fp": int(len(match["fp_probs"])), "area_mm2": float(match["area_mm2"]),
    }


def slide_froc(match: dict, *, fp_points: list[int] | None = None) -> dict:
    """Per-slide FROC with the official guard: no predictions scores 0.

    The official ``get_froc_vals_pr`` returns ``froc_score_slide = 0`` when a
    slide has no predicted points for the class, rather than falling through to
    the single-value fallback used only when pooling slides.
    """
    fp_points = fp_points or FP_POINTS
    if int(match.get("n_pred", 0)) == 0:
        return _zero_froc(match, fp_points)
    return froc_from_match(match, fp_points=fp_points)


def froc_score(pred_points: np.ndarray, gt_points: np.ndarray,
               area_mm2: float, mpp: float, *,
               class_name: str = "inflammatory",
               fp_points: list[int] | None = None,
               match_radius_um: dict | None = None) -> dict:
    """Mean sensitivity at the challenge FP/mm2 operating points for one case.

    ``pred_points`` and ``gt_points`` are ``[N, 4]`` / ``[M, 4]`` arrays in
    level-0 pixels. ``fp_points`` defaults to :data:`config.FP_POINTS` and
    ``match_radius_um`` to :data:`config.MATCH_RADIUS_UM`. Returns a dict with
    ``"froc"``, ``"sensitivities"`` (per FP/mm2 point) and the raw curve. This
    is the per-slide score, so a case with no predictions scores 0.
    """
    match = case_match(pred_points, gt_points, area_mm2, mpp,
                       class_name=class_name, match_radius_um=match_radius_um)
    return slide_froc(match, fp_points=fp_points or FP_POINTS)


def froc_curve(pred_points: np.ndarray, gt_points: np.ndarray,
               area_mm2: float, mpp: float, *,
               class_name: str = "inflammatory",
               match_radius_um: dict | None = None) -> dict:
    """Full sensitivity vs FP/mm2 curve for plotting.

    Returns ``{"fp_per_mm2": np.ndarray, "sensitivity": np.ndarray}`` ordered
    by descending score threshold (as produced by the MONAI curve sweep).
    """
    match = case_match(pred_points, gt_points, area_mm2, mpp,
                       class_name=class_name, match_radius_um=match_radius_um)
    if match["total_pos"] == 0 or match["area_mm2"] <= 0:
        return {"fp_per_mm2": np.zeros(1), "sensitivity": np.zeros(1)}
    fp_per_mm2, sensitivity = froc_curve_data(
        match["fp_probs"], match["tp_probs"], match["total_pos"],
        match["area_mm2"])
    return {"fp_per_mm2": fp_per_mm2, "sensitivity": sensitivity}
