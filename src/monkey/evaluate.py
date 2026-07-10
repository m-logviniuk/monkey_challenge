"""Pooled out-of-fold (leave-one-centre-out) evaluation.

Each case is scored by the fold that held its centre out, so no case is scored
by a model that trained on its centre (no leakage). For every held-out case the
predictions are matched to ground truth inside the ROI; the matched/unmatched
scores are pooled across cases and the FROC score, sensitivities, and
precision/recall are computed on the pooled set - the reporting standard for
this project (pooled OOF, not noisy per-fold headline values).

Confidence is a BCa bootstrap over cases (2000 resamples) giving a 95% CI and
SD for the FROC score, reported per class (lymphocyte, monocyte), combined
(inflammatory), and per centre. Results are written to ``results/metrics.json``.

``monkey.cli`` calls :func:`evaluate_oof` for the ``oof`` subcommand. It builds
on ``monkey.detect.detect_case``, ``monkey.data.read_case_index`` /
``leave_one_centre_out_folds``, ``monkey.checkpoint.load_fold_model``, and the
scoring in :mod:`monkey.froc` and :mod:`monkey.metrics`.

metrics.json schema (all FROC values in [0, 1], sensitivities keyed by FP/mm2):
    meta        run configuration (tag, radii, FP points, thresholds, seed,
                bootstrap count, aggregation description).
    overall     per class in {lymphocyte, monocyte, inflammatory}:
                froc, froc_ci [lo, hi], froc_sd, sensitivities {fp: sens},
                precision_recall {"0.4"/"0.9": {precision, recall, f1, ...}},
                total_pos, n_tp, n_fp, area_mm2, n_cases, and the compact
                curve / pr_curve / calibration arrays used by the figures.
    per_centre  same block per centre {A, B, C, D}, plus n_cases.
    per_case    per case: centre and per-class {froc, total_pos, n_tp, n_fp}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import (
    CENTERS,
    CLASS_NAMES,
    FP_POINTS,
    MATCH_RADIUS_UM,
    PROB_THRESHOLDS,
    RESULTS_DIR,
    SEED,
    config_tag,
)
from .froc import case_match, froc_curve_data, froc_from_match, slide_froc
from .metrics import (
    bca_ci,
    pr_curve,
    precision_recall_from_counts,
    reliability,
)

SCORED_CLASSES = (*CLASS_NAMES, "inflammatory")


def _gt_level0(case) -> np.ndarray:
    """Ground-truth points as ``[M, 4]`` (x0, y0, score, class) in level-0."""
    pts = case.points
    if pts is None or len(pts) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = np.zeros((len(pts), 4), dtype=np.float32)
    for i, (pi, x, y, cls) in enumerate(pts):
        x0, y0 = case.patch_xy[int(pi)]
        out[i] = (x0 + x * case.downsample, y0 + y * case.downsample, 1.0, cls)
    return out


def _roi_area_mm2(case) -> float:
    """Scored ROI area in mm2 from ROI-mask coverage at the patch scale."""
    px = float(case.roi_mask.sum())
    return px * (case.mpp / 1000.0) ** 2


def _pool(records: list[dict]) -> dict:
    """Concatenate per-case match dicts into one pooled match dict."""
    if not records:
        return {"tp_probs": np.zeros(0), "fp_probs": np.zeros(0),
                "total_pos": 0, "area_mm2": 0.0}
    return {
        "tp_probs": np.concatenate([r["tp_probs"] for r in records]),
        "fp_probs": np.concatenate([r["fp_probs"] for r in records]),
        "total_pos": int(sum(r["total_pos"] for r in records)),
        "area_mm2": float(sum(r["area_mm2"] for r in records)),
    }


def _summarise(records: list[dict], prob_thresholds: list[float],
               fp_points: list[int], *, with_ci: bool, seed: int) -> dict:
    """Pooled FROC + precision/recall (+ BCa CI/SD) for a list of case matches."""
    pooled = _pool(records)
    froc = froc_from_match(pooled, fp_points=fp_points)
    out = {
        "froc": froc["froc"],
        "sensitivities": froc["sensitivities"],
        "total_pos": froc["total_pos"],
        "n_tp": froc["n_tp"],
        "n_fp": froc["n_fp"],
        "area_mm2": froc["area_mm2"],
        "n_cases": len(records),
    }
    pr = {}
    for thr in prob_thresholds:
        pr[str(thr)] = precision_recall_from_counts(
            pooled["tp_probs"], pooled["fp_probs"], pooled["total_pos"], thr)
    out["precision_recall"] = pr

    # Compact curves for the figure scripts (avoids re-running detection).
    if pooled["total_pos"] > 0 and pooled["area_mm2"] > 0:
        fp_curve, sens_curve = froc_curve_data(
            pooled["fp_probs"], pooled["tp_probs"], pooled["total_pos"],
            pooled["area_mm2"])
        out["curve"] = {"fp_per_mm2": fp_curve.tolist(),
                        "sensitivity": sens_curve.tolist()}
    else:
        out["curve"] = {"fp_per_mm2": [0.0], "sensitivity": [0.0]}
    out["pr_curve"] = pr_curve(pooled["tp_probs"], pooled["fp_probs"],
                               pooled["total_pos"])
    out["calibration"] = reliability(pooled["tp_probs"], pooled["fp_probs"])

    if with_ci and len(records) >= 2:
        ci = bca_ci(records,
                    lambda s: froc_from_match(_pool(list(s)),
                                              fp_points=fp_points)["froc"],
                    seed=seed)
        out["froc_ci"] = [ci["lo"], ci["hi"]]
        out["froc_sd"] = ci["sd"]
    else:
        out["froc_ci"] = [out["froc"], out["froc"]]
        out["froc_sd"] = 0.0
    return out


def evaluate_oof(data_dir=None, ckpt_dir=None, device=None, out_path=None, *,
                 tag: str | None = None,
                 prob_thresholds: list[float] | None = None,
                 fp_points: list[int] | None = None,
                 max_cases: int | None = None) -> dict:
    """Pooled out-of-fold LOCO evaluation; writes ``metrics.json``.

    ``tag`` selects the checkpoint namespace (see :func:`config.config_tag`).
    ``max_cases`` optionally caps the cases per fold (smoke use). Returns the
    summary dict that is also written to ``out_path`` (default
    ``results/metrics.json``).
    """
    import torch
    from tqdm import tqdm

    from .checkpoint import load_fold_model
    from .config import DATA_DIR
    from .data import (
        MonkeyCase,
        leave_one_centre_out_folds,
        read_case_index,
    )
    from .detect import detect_case

    data_dir = Path(data_dir) if data_dir else DATA_DIR
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = tag if tag is not None else config_tag()
    prob_thresholds = prob_thresholds or PROB_THRESHOLDS
    fp_points = fp_points or FP_POINTS

    case_index = read_case_index(data_dir)
    folds = leave_one_centre_out_folds(case_index)
    if not folds:
        raise RuntimeError(f"no cases found under {data_dir}.")

    # per-case match records, keyed by scored class name
    records: dict[str, list[dict]] = {c: [] for c in SCORED_CLASSES}
    centre_of: dict[str, str] = {}
    per_case: dict[str, dict] = {}

    for fold_i, fold in enumerate(folds, start=1):
        model = load_fold_model(fold_i, ckpt_dir=ckpt_dir, tag=tag,
                                device=device)
        val = fold["val"][:max_cases] if max_cases else fold["val"]
        for case_name, centre, path in tqdm(
                val, desc=f"oof fold {fold_i} (centre {fold['val_center']})"):
            case = MonkeyCase(path)
            pred = detect_case(model, case, device, prob_threshold=0.0)
            gt = _gt_level0(case)
            area = _roi_area_mm2(case)
            centre_of[case_name] = centre
            per_case[case_name] = {"centre": centre}
            for cls in SCORED_CLASSES:
                m = case_match(pred, gt, area, case.base_mpp, class_name=cls,
                               match_radius_um=MATCH_RADIUS_UM)
                m["case"] = case_name
                m["centre"] = centre
                records[cls].append(m)
                fr = slide_froc(m, fp_points=fp_points)
                per_case[case_name][cls] = {
                    "froc": fr["froc"], "total_pos": fr["total_pos"],
                    "n_tp": fr["n_tp"], "n_fp": fr["n_fp"]}

    overall = {
        cls: _summarise(records[cls], prob_thresholds, fp_points,
                        with_ci=True, seed=SEED)
        for cls in SCORED_CLASSES
    }

    per_centre: dict[str, dict] = {}
    centres = [c for c in CENTERS if any(v == c for v in centre_of.values())]
    for centre in centres:
        block = {}
        for cls in SCORED_CLASSES:
            recs = [r for r in records[cls] if r["centre"] == centre]
            block[cls] = _summarise(recs, prob_thresholds, fp_points,
                                    with_ci=True, seed=SEED)
        block["n_cases"] = sum(1 for v in centre_of.values() if v == centre)
        per_centre[centre] = block

    summary = {
        "meta": {
            "tag": tag,
            "n_cases": len(per_case),
            "centres": centres,
            "fp_points": fp_points,
            "prob_thresholds": prob_thresholds,
            "match_radius_um": MATCH_RADIUS_UM,
            "bootstrap_resamples": 2000,
            "seed": SEED,
            "aggregation": "pooled out-of-fold (leave-one-centre-out)",
            "froc_note": ("reproduces monai.metrics FROC used by the official "
                          "MONKEY evaluation; matching mirrors the official "
                          "nearest-neighbour rule"),
        },
        "overall": overall,
        "per_centre": per_centre,
        "per_case": per_case,
    }

    out_path = Path(out_path) if out_path else (RESULTS_DIR / "metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    infl = overall["inflammatory"]
    print(f"pooled OOF FROC (inflammatory): {infl['froc']:.3f} "
          f"[{infl['froc_ci'][0]:.3f}, {infl['froc_ci'][1]:.3f}] "
          f"sd {infl['froc_sd']:.3f} over {len(per_case)} cases")
    for cls in CLASS_NAMES:
        c = overall[cls]
        print(f"  {cls}: froc={c['froc']:.3f} "
              f"[{c['froc_ci'][0]:.3f}, {c['froc_ci'][1]:.3f}]")
    print(f"metrics written: {out_path}")
    return summary
