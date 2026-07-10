"""Exploratory data-analysis figures for the MONKEY cohort.

Reads the per-case HDF5 files directly (no model, no torch) and summarises the
cohort: cases per centre, annotated cells per centre split by class,
lymphocyte/monocyte class balance, and the cells-per-mm2 density distribution.
The ROI area is taken from the ROI-mask coverage at the patch scale, the same
area used to normalise false positives in the FROC.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from figstyle import CENTER_COLORS, CLASS_COLORS, apply_style, data_dir, save  # noqa: E402
from monkey.config import BASE_MPP_DEFAULT, CENTERS, DOWNSAMPLE_DEFAULT  # noqa: E402


def _iter_cases(directory: Path):
    manifest = directory / "monkey_manifest.csv"
    if manifest.exists():
        import csv
        with open(manifest, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                path = directory / f"{row['case']}.h5"
                if path.exists():
                    yield row["case"], row["center"], path
    else:
        for path in sorted(directory.glob("*.h5")):
            with h5py.File(path, "r") as f:
                center = str(f.attrs.get("center", "?"))
                case = str(f.attrs.get("case", path.stem))
            yield case, center, path


def collect(directory: Path) -> list[dict]:
    """Per-case cohort statistics."""
    rows = []
    for case, center, path in _iter_cases(directory):
        with h5py.File(path, "r") as f:
            roi_px = float(np.asarray(f["roi_mask"]).sum())
            points = np.asarray(f["points"])
            downsample = float(f.attrs.get("downsample", DOWNSAMPLE_DEFAULT))
            base_mpp = float(f.attrs.get("base_mpp", BASE_MPP_DEFAULT))
        mpp = base_mpp * downsample
        area_mm2 = roi_px * (mpp / 1000.0) ** 2
        cls = points[:, 3].astype(int) if len(points) else np.zeros(0, dtype=int)
        rows.append({
            "case": case, "center": center, "area_mm2": area_mm2,
            "n_lymphocyte": int((cls == 0).sum()),
            "n_monocyte": int((cls == 1).sum()),
            "n_cells": int(len(cls)),
        })
    return rows


def figure(rows: list[dict], out_dir: Path):
    apply_style()
    centres = [c for c in CENTERS if any(r["center"] == c for r in rows)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("MONKEY cohort overview (PAS kidney biopsies)", fontsize=14)

    # (a) cases per centre
    ax = axes[0, 0]
    counts = [sum(1 for r in rows if r["center"] == c) for c in centres]
    ax.bar(centres, counts, color=[CENTER_COLORS.get(c, "gray") for c in centres])
    for i, v in enumerate(counts):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("cases")
    ax.set_ylim(0, max(counts) * 1.18 if counts else 1)
    ax.set_title("cases per centre")

    # (b) cells per centre, split by class
    ax = axes[0, 1]
    lymph = [sum(r["n_lymphocyte"] for r in rows if r["center"] == c)
             for c in centres]
    mono = [sum(r["n_monocyte"] for r in rows if r["center"] == c)
            for c in centres]
    ax.bar(centres, lymph, label="lymphocyte", color=CLASS_COLORS["lymphocyte"])
    ax.bar(centres, mono, bottom=lymph, label="monocyte",
           color=CLASS_COLORS["monocyte"])
    ax.set_ylabel("annotated cells")
    ax.set_title("cells per centre by class")
    ax.legend()

    # (c) class balance overall
    ax = axes[1, 0]
    total_l = sum(r["n_lymphocyte"] for r in rows)
    total_m = sum(r["n_monocyte"] for r in rows)
    ax.bar(["lymphocyte", "monocyte"], [total_l, total_m],
           color=[CLASS_COLORS["lymphocyte"], CLASS_COLORS["monocyte"]])
    total = max(total_l + total_m, 1)
    for i, v in enumerate([total_l, total_m]):
        ax.text(i, v, f"{v}\n({100 * v / total:.0f}%)", ha="center", va="bottom")
    ax.set_ylabel("annotated cells")
    ax.set_ylim(0, max(total_l, total_m) * 1.25)
    ax.set_title("class balance (whole cohort)")

    # (d) cells per mm2 distribution, per centre
    ax = axes[1, 1]
    data = []
    for c in centres:
        vals = [r["n_cells"] / r["area_mm2"] for r in rows
                if r["center"] == c and r["area_mm2"] > 0]
        data.append(vals)
    bp = ax.boxplot(data, labels=centres, patch_artist=True)
    for patch, c in zip(bp["boxes"], centres, strict=True):
        patch.set_facecolor(CENTER_COLORS.get(c, "gray"))
        patch.set_alpha(0.7)
    ax.set_ylabel("cells per mm2 (ROI)")
    ax.set_title("inflammatory-cell density per centre")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out_dir, "fig_eda_cohort.png")


def main():
    directory = data_dir()
    out_dir = Path(__import__("figstyle").results_dir())
    print(f"EDA: reading cases from {directory}")
    rows = collect(directory)
    if not rows:
        print(f"no .h5 cases under {directory}; set MONKEY_DATA_DIR.")
        return
    figure(rows, out_dir)
    summary = {
        "n_cases": len(rows),
        "n_cells": sum(r["n_cells"] for r in rows),
        "per_centre": {
            c: {"cases": sum(1 for r in rows if r["center"] == c),
                "cells": sum(r["n_cells"] for r in rows if r["center"] == c)}
            for c in CENTERS if any(r["center"] == c for r in rows)},
    }
    with open(out_dir / "eda_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"eda_summary.json ({summary['n_cases']} cases, "
          f"{summary['n_cells']} cells)")


if __name__ == "__main__":
    main()
