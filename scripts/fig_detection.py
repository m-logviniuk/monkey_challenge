"""Detection-interpretability figures on example ROIs, per centre.

For each centre a held-out case is scored by the fold that trained without that
centre (a genuine out-of-fold prediction). On the tile with the most annotated
cells the script shows: the PAS patch with ground-truth dots, the predicted
density heatmap with predicted dots, and the true/false positive/negative
breakdown from the challenge matching. Needs the packed HDF5 cases and the fold
checkpoints, so it runs on Colab (imports torch).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from figstyle import apply_style, data_dir, results_dir, save  # noqa: E402
from monkey.checkpoint import load_fold_model  # noqa: E402
from monkey.config import CLASS_NAMES, MATCH_RADIUS_UM  # noqa: E402
from monkey.data import MonkeyCase, read_case_index, to_model_input  # noqa: E402
from monkey.data import leave_one_centre_out_folds, tile_points  # noqa: E402
from monkey.detect import peaks_from_density  # noqa: E402
from monkey.froc import match_points  # noqa: E402

DISPLAY_THRESHOLD = 0.5


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _denormalise(image_chw: np.ndarray) -> np.ndarray:
    """Undo the ImageNet normalisation for display; returns HxWx3 in [0, 1]."""
    from monkey.config import IMAGENET_MEAN, IMAGENET_STD
    mean = np.array(IMAGENET_MEAN)[:, None, None]
    std = np.array(IMAGENET_STD)[:, None, None]
    img = image_chw * std + mean
    return np.clip(np.transpose(img, (1, 2, 0)), 0, 1)


def _best_tile(case: MonkeyCase) -> int:
    """Tile index with the most annotated cells (ties -> first)."""
    if len(case.points) == 0:
        return 0
    idx = case.points[:, 0].astype(int)
    counts = np.bincount(idx, minlength=case.n_tiles)
    return int(np.argmax(counts))


def _radius_px(case: MonkeyCase) -> dict:
    return {0: MATCH_RADIUS_UM[CLASS_NAMES[0]] / case.mpp,
            1: MATCH_RADIUS_UM[CLASS_NAMES[1]] / case.mpp}


def _panels(ax_row, case: MonkeyCase, tile: int, model, device):
    patch = case.patches[tile]
    roi = case.roi_mask[tile].astype(np.float32)
    x = to_model_input(patch).to(device)
    with torch.no_grad():
        density = torch.sigmoid(model(x)).cpu().numpy()[0] * roi[None]
    img = _denormalise(to_model_input(patch).cpu().numpy()[0])

    gt = tile_points(case.points, tile)          # (x, y, class)
    peaks = peaks_from_density(density, DISPLAY_THRESHOLD, _radius_px(case))

    # panel A: PAS + GT dots
    ax = ax_row[0]
    ax.imshow(img)
    for cls, col in ((0, "lime"), (1, "yellow")):
        sel = gt[gt[:, 2] == cls]
        ax.scatter(sel[:, 0], sel[:, 1], s=14, c=col, edgecolors="k",
                   linewidths=0.3, label=CLASS_NAMES[cls])
    ax.set_title(f"{case.center} / {case.case}\nPAS + ground-truth dots",
                 fontsize=9)
    ax.axis("off")
    ax.legend(fontsize=6, loc="upper right")

    # panel B: predicted density + predicted dots
    ax = ax_row[1]
    ax.imshow(img)
    ax.imshow(density.max(axis=0), cmap="hot", alpha=0.5, vmin=0, vmax=1)
    if len(peaks):
        ax.scatter(peaks[:, 0], peaks[:, 1], s=12, facecolors="none",
                   edgecolors="cyan", linewidths=0.8)
    ax.set_title(f"predicted density + peaks (>{DISPLAY_THRESHOLD})", fontsize=9)
    ax.axis("off")

    # panel C: TP / FP / FN using the challenge matching
    ax = ax_row[2]
    ax.imshow(img)
    tp = fp = fn = 0
    for cls in (0, 1):
        gt_c = gt[gt[:, 2] == cls][:, :2]
        pk_c = peaks[peaks[:, 3] == cls][:, :2] if len(peaks) else np.zeros((0, 2))
        r = _radius_px(case)[cls]
        tp_mask, gt_mask = match_points(pk_c, gt_c, r)
        if len(pk_c):
            ax.scatter(pk_c[tp_mask, 0], pk_c[tp_mask, 1], s=16, c="lime",
                       marker="o")
            ax.scatter(pk_c[~tp_mask, 0], pk_c[~tp_mask, 1], s=16, c="red",
                       marker="x")
        if len(gt_c):
            miss = gt_c[~gt_mask]
            ax.scatter(miss[:, 0], miss[:, 1], s=24, facecolors="none",
                       edgecolors="deepskyblue", linewidths=1.0)
        tp += int(tp_mask.sum())
        fp += int((~tp_mask).sum()) if len(pk_c) else 0
        fn += int((~gt_mask).sum()) if len(gt_c) else 0
    ax.set_title(f"TP={tp} (green)  FP={fp} (red x)  FN={fn} (blue o)",
                 fontsize=9)
    ax.axis("off")


def main():
    apply_style()
    directory = data_dir()
    ckpt_dir = None  # falls back to MONKEY_CKPT_DIR / config default
    index = read_case_index(directory)
    folds = leave_one_centre_out_folds(index)
    if not folds:
        print(f"no cases under {directory}; set MONKEY_DATA_DIR.")
        return
    device = _device()

    rows = []
    for fold_i, fold in enumerate(folds, start=1):
        val = fold["val"]
        if not val:
            continue
        # example case = the held-out case with the most annotated cells
        best_case, best_n = None, -1
        for _, _, path in val:
            with __import__("h5py").File(path, "r") as f:
                n = int(f["points"].shape[0])
            if n > best_n:
                best_n, best_case = n, path
        rows.append((fold_i, fold["val_center"], best_case))

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[None, :]
    fig.suptitle("Out-of-fold detection on example ROIs, per centre",
                 fontsize=14)
    for r, (fold_i, centre, path) in enumerate(rows):
        model = load_fold_model(fold_i, ckpt_dir=ckpt_dir, device=device)
        case = MonkeyCase(path)
        _panels(axes[r], case, _best_tile(case), model, device)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, results_dir(), "fig_detection_overlay.png")


if __name__ == "__main__":
    main()
