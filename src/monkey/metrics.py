"""Detection metrics: precision/recall at fixed thresholds and BCa bootstrap.

Precision/recall reproduce the official MONKEY
``compute_precision_recall_threshold``: predictions are first matched to
ground truth (nearest-neighbour, :func:`monkey.froc.match_coordinates`), then
the matched (``tp_probs``) and unmatched (``fp_probs``) scores are thresholded
with a strict ``>`` and counted, with ``fn = total_pos - tp``.

Confidence intervals use the bias-corrected and accelerated (BCa) bootstrap
over cases/slides, which is the reporting standard adopted for this project
(pooled out-of-fold estimate + BCa 95% CI + SD). The generic :func:`bca_ci`
takes per-case records and a pooling ``statistic`` so it works for the pooled
FROC score (not just a mean of per-case values); :func:`bootstrap_ci` is the
special case of the mean of scalar values.

Coordinate and unit conventions match :mod:`monkey.froc`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import norm

from .config import BOOTSTRAP_RESAMPLES, MATCH_RADIUS_UM, SEED
from .froc import case_match


def precision_recall_from_counts(tp_probs: np.ndarray, fp_probs: np.ndarray,
                                 total_pos: int, threshold: float) -> dict:
    """Precision/recall/F1 at a score ``threshold`` from matched scores.

    Mirrors the official ``compute_precision_recall_threshold``: the score
    filter is a strict ``>`` and ``fn = total_pos - tp``.
    """
    tp_probs = np.asarray(tp_probs, dtype=np.float64)
    fp_probs = np.asarray(fp_probs, dtype=np.float64)
    y_score = np.concatenate([tp_probs, fp_probs])
    y_true = np.concatenate([np.ones(len(tp_probs)), np.zeros(len(fp_probs))])
    keep = y_score > threshold
    tp = int(y_true[keep].sum())
    fp = int(keep.sum()) - tp
    fn = int(total_pos) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": float(precision), "recall": float(recall),
            "f1": float(f1), "tp": tp, "fp": fp, "fn": fn}


def precision_recall(pred_points: np.ndarray, gt_points: np.ndarray,
                     mpp: float, prob_threshold: float, *,
                     class_name: str = "inflammatory",
                     match_radius_um: dict | None = None) -> dict:
    """Precision, recall, and F1 at a probability threshold for one case.

    Predictions are matched to ground truth within the class radius; the
    matched/unmatched scores are then thresholded. Returns
    ``{"precision", "recall", "f1", "tp", "fp", "fn"}``.
    """
    match = case_match(pred_points, gt_points, 1.0, mpp, class_name=class_name,
                       match_radius_um=match_radius_um or MATCH_RADIUS_UM)
    return precision_recall_from_counts(match["tp_probs"], match["fp_probs"],
                                        match["total_pos"], prob_threshold)


def pr_curve(tp_probs: np.ndarray, fp_probs: np.ndarray, total_pos: int,
             thresholds: np.ndarray | None = None) -> dict:
    """Precision/recall across a grid of score thresholds (for PR plots).

    Returns ``{"threshold", "precision", "recall"}`` as lists, evaluated with
    the same strict ``>`` rule as :func:`precision_recall_from_counts`.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 51)
    prec, rec = [], []
    for thr in thresholds:
        pr = precision_recall_from_counts(tp_probs, fp_probs, total_pos, thr)
        prec.append(pr["precision"])
        rec.append(pr["recall"])
    return {"threshold": [float(t) for t in thresholds],
            "precision": prec, "recall": rec}


def reliability(tp_probs: np.ndarray, fp_probs: np.ndarray,
                n_bins: int = 10) -> dict:
    """Calibration: empirical precision of predictions per score bin.

    Each predicted point is a true positive (from ``tp_probs``) or false
    positive (from ``fp_probs``); within a score bin the empirical precision is
    the true-positive fraction. Returns ``{"bin_mid", "precision", "count"}``.
    """
    tp_probs = np.asarray(tp_probs, dtype=np.float64)
    fp_probs = np.asarray(fp_probs, dtype=np.float64)
    scores = np.concatenate([tp_probs, fp_probs])
    labels = np.concatenate([np.ones(len(tp_probs)), np.zeros(len(fp_probs))])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids, precs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (scores >= lo) & (scores < hi) if hi < 1.0 else \
            (scores >= lo) & (scores <= hi)
        n = int(sel.sum())
        mids.append(float((lo + hi) / 2.0))
        precs.append(float(labels[sel].mean()) if n > 0 else float("nan"))
        counts.append(n)
    return {"bin_mid": mids, "precision": precs, "count": counts}


def _bca_endpoints(theta_hat: float, boot: np.ndarray, jack: np.ndarray,
                   alpha: float) -> tuple[float, float]:
    """BCa lower/upper percentile probabilities from bootstrap + jackknife."""
    boot = np.asarray(boot, dtype=np.float64)
    prop = float(np.mean(boot < theta_hat))
    if prop <= 0.0 or prop >= 1.0:
        # No bias correction possible; fall back to the percentile interval.
        return alpha / 2.0, 1.0 - alpha / 2.0
    z0 = norm.ppf(prop)

    jack = np.asarray(jack, dtype=np.float64)
    jack_mean = jack.mean()
    diff = jack_mean - jack
    denom = 6.0 * (np.sum(diff ** 2) ** 1.5)
    acc = float(np.sum(diff ** 3) / denom) if denom > 0 else 0.0

    z_lo, z_hi = norm.ppf(alpha / 2.0), norm.ppf(1.0 - alpha / 2.0)

    def adjust(z: float) -> float:
        return float(norm.cdf(z0 + (z0 + z) / (1.0 - acc * (z0 + z))))

    return adjust(z_lo), adjust(z_hi)


def bca_ci(records: Sequence, statistic: Callable[[Sequence], float],
           n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = SEED,
           alpha: float = 0.05) -> dict:
    """BCa bootstrap over ``records`` for a pooling ``statistic``.

    ``records`` is a sequence of per-case items and ``statistic`` maps a list of
    such items to a scalar (e.g. pool the per-case matches and compute FROC).
    Resampling is over records (cases/slides). Returns
    ``{"point", "lo", "hi", "sd", "n"}`` where ``point`` is the estimate on the
    full sample and ``sd`` is the bootstrap standard deviation.
    """
    records = list(records)
    n = len(records)
    point = float(statistic(records))
    if n < 2:
        return {"point": point, "lo": point, "hi": point, "sd": 0.0, "n": n}

    rng = np.random.RandomState(seed)
    boot = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boot[b] = statistic([records[i] for i in idx])

    jack = np.empty(n, dtype=np.float64)
    for i in range(n):
        jack[i] = statistic([records[j] for j in range(n) if j != i])

    lo_p, hi_p = _bca_endpoints(point, boot, jack, alpha)
    lo = float(np.percentile(boot, 100.0 * lo_p))
    hi = float(np.percentile(boot, 100.0 * hi_p))
    return {"point": point, "lo": lo, "hi": hi,
            "sd": float(boot.std(ddof=1)), "n": n}


def bootstrap_ci(values: list[float], n_resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = SEED, alpha: float = 0.05) -> tuple[float, float]:
    """BCa bootstrap confidence interval on the mean of ``values``."""
    vals = np.asarray(values, dtype=np.float64)
    res = bca_ci(list(vals), lambda s: float(np.mean(s)) if len(s) else 0.0,
                 n_resamples=n_resamples, seed=seed, alpha=alpha)
    return res["lo"], res["hi"]
