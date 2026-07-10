"""FROC, precision-recall, and calibration figures from metrics.json.

Reads the pooled out-of-fold results written by ``monkey oof`` and plots the
challenge FROC curves (per class and per centre), the precision-recall curves,
and a calibration (reliability) plot. No data or model is needed; everything is
taken from the compact arrays stored in metrics.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402

from figstyle import (  # noqa: E402
    CENTER_COLORS,
    CLASS_COLORS,
    FP_POINTS,
    apply_style,
    metrics_path,
    results_dir,
    save,
)

CLASSES = ("lymphocyte", "monocyte", "inflammatory")


def _load() -> dict | None:
    path = metrics_path()
    if not path.exists():
        print(f"metrics.json not found at {path}; run `monkey oof` first.")
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _sens_points(block: dict):
    """Operating-point (FP/mm2 -> sensitivity) pairs from a class block."""
    sens = block.get("sensitivities", {})
    xs, ys = [], []
    for p in FP_POINTS:
        key = str(p) if str(p) in sens else p
        if key in sens:
            xs.append(p)
            ys.append(sens[key])
    return xs, ys


def _plot_froc(ax, block: dict, color: str, label: str):
    curve = block.get("curve", {})
    fp = curve.get("fp_per_mm2", [])
    sens = curve.get("sensitivity", [])
    if fp:
        ax.plot(fp, sens, color=color, lw=1.8, alpha=0.9, label=label)
    xs, ys = _sens_points(block)
    ax.scatter(xs, ys, color=color, s=18, zorder=3)


def fig_froc_curves(metrics: dict, out_dir: Path):
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("FROC: sensitivity vs false positives per mm2 "
                 "(pooled out-of-fold)", fontsize=13)

    overall = metrics["overall"]
    for cls in CLASSES:
        block = overall.get(cls)
        if not block:
            continue
        froc = block["froc"]
        lo, hi = block.get("froc_ci", [froc, froc])
        _plot_froc(ax1, block, CLASS_COLORS[cls],
                   f"{cls} (FROC={froc:.3f} [{lo:.3f},{hi:.3f}])")
    ax1.set_xlim(0, max(FP_POINTS) * 1.05)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("false positives per mm2")
    ax1.set_ylabel("sensitivity")
    ax1.set_title("per class")
    ax1.legend(fontsize=8, loc="lower right")

    per_centre = metrics.get("per_centre", {})
    for centre, block in per_centre.items():
        infl = block.get("inflammatory")
        if not infl:
            continue
        _plot_froc(ax2, infl, CENTER_COLORS.get(centre, "gray"),
                   f"centre {centre} (FROC={infl['froc']:.3f})")
    ax2.set_xlim(0, max(FP_POINTS) * 1.05)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("false positives per mm2")
    ax2.set_ylabel("sensitivity")
    ax2.set_title("inflammatory, per centre")
    ax2.legend(fontsize=8, loc="lower right")

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out_dir, "fig_froc_curves.png")


def fig_precision_recall(metrics: dict, out_dir: Path):
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.suptitle("Precision-recall (pooled out-of-fold)", fontsize=13)
    overall = metrics["overall"]
    for cls in CLASSES:
        block = overall.get(cls)
        pr = block.get("pr_curve") if block else None
        if not pr:
            continue
        ax.plot(pr["recall"], pr["precision"], color=CLASS_COLORS[cls],
                lw=1.8, label=cls)
        # mark the two reporting thresholds
        for thr in ("0.4", "0.9"):
            d = block.get("precision_recall", {}).get(thr)
            if d:
                ax.scatter(d["recall"], d["precision"],
                           color=CLASS_COLORS[cls], s=25, zorder=3)
                ax.annotate(thr, (d["recall"], d["precision"]),
                            fontsize=7, textcoords="offset points",
                            xytext=(3, 3))
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out_dir, "fig_precision_recall.png")


def fig_calibration(metrics: dict, out_dir: Path):
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.suptitle("Calibration: empirical precision by score bin", fontsize=13)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="ideal")
    overall = metrics["overall"]
    for cls in CLASSES:
        block = overall.get(cls)
        cal = block.get("calibration") if block else None
        if not cal:
            continue
        mids = cal["bin_mid"]
        prec = cal["precision"]
        xs = [m for m, p in zip(mids, prec, strict=True) if p == p]  # drop NaN
        ys = [p for p in prec if p == p]
        ax.plot(xs, ys, marker="o", ms=4, color=CLASS_COLORS[cls], label=cls)
    ax.set_xlabel("predicted score")
    ax.set_ylabel("empirical precision (true-positive fraction)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out_dir, "fig_calibration.png")


def main():
    metrics = _load()
    if metrics is None:
        return
    out_dir = results_dir()
    fig_froc_curves(metrics, out_dir)
    fig_precision_recall(metrics, out_dir)
    fig_calibration(metrics, out_dir)


if __name__ == "__main__":
    main()
