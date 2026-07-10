"""Density map to discrete cell detections.

The model's sigmoid density map is turned into predicted points by finding
local maxima above a probability threshold and suppressing near-duplicates
within the class matching radius (non-maximum suppression). Predicted points
are mapped back to level-0 slide pixels using the tile origin and downsample,
which is the coordinate space the challenge scores in.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import maximum_filter

from .config import MATCH_RADIUS_UM


def _greedy_nms(coords: np.ndarray, scores: np.ndarray,
                radius: float) -> np.ndarray:
    """Return indices kept after greedy NMS on 2D coords within ``radius``."""
    order = np.argsort(-scores)
    kept: list[int] = []
    taken = np.zeros(len(coords), dtype=bool)
    r2 = radius * radius
    for i in order:
        if taken[i]:
            continue
        kept.append(i)
        dx = coords[:, 0] - coords[i, 0]
        dy = coords[:, 1] - coords[i, 1]
        taken |= (dx * dx + dy * dy) <= r2
    return np.asarray(kept, dtype=int)


def peaks_from_density(density: np.ndarray, prob_threshold: float,
                       radius_px: dict[int, float]) -> np.ndarray:
    """Local-maxima peak detection with per-class NMS.

    ``density`` is a ``[n_classes, H, W]`` map in [0, 1]. ``radius_px`` maps a
    class index to its NMS radius in pixels. Returns an array with columns
    ``(x, y, score, class)`` in patch pixels.
    """
    out = []
    for cls in range(density.shape[0]):
        dmap = density[cls]
        r = max(1, int(round(radius_px[cls])))
        window = 2 * r + 1
        local_max = maximum_filter(dmap, size=window, mode="constant")
        mask = (dmap == local_max) & (dmap > prob_threshold)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        scores = dmap[ys, xs]
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        keep = _greedy_nms(coords, scores, float(r))
        for k in keep:
            out.append((float(xs[k]), float(ys[k]), float(scores[k]), cls))
    if not out:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def _class_radius_px(mpp: float) -> dict[int, float]:
    # Class order matches config.CLASS_NAMES: 0 lymphocyte, 1 monocyte.
    return {
        0: MATCH_RADIUS_UM["lymphocyte"] / mpp,
        1: MATCH_RADIUS_UM["monocyte"] / mpp,
    }


@torch.no_grad()
def detect_case(model, case_h5, device, prob_threshold: float = 0.0,
                batch_size: int = 16) -> np.ndarray:
    """Detect cells in one case and return level-0 predicted points.

    Runs the model over every tile of ``case_h5`` (a path or an open
    :class:`~monkey.data.MonkeyCase`), detects per-class peaks inside the ROI,
    and maps them to level-0 slide pixels.

    Returns an ``np.ndarray`` of shape ``[N, 4]`` with columns
    ``(x0, y0, score, class)`` in level-0 pixels. ``class`` is 0 for
    lymphocyte and 1 for monocyte.
    """
    from .data import MonkeyCase, to_model_input

    case = case_h5 if isinstance(case_h5, MonkeyCase) else MonkeyCase(case_h5)
    radius_px = _class_radius_px(case.mpp)
    model.eval()

    detections: list[np.ndarray] = []
    n_tiles = case.n_tiles
    for start in range(0, n_tiles, batch_size):
        stop = min(start + batch_size, n_tiles)
        patches = case.patches[start:stop]
        rois = case.roi_mask[start:stop]
        x = to_model_input(patches).to(device)
        probs = torch.sigmoid(model(x)).cpu().numpy()
        for j in range(stop - start):
            idx = start + j
            density = probs[j] * rois[j].astype(np.float32)[None]
            peaks = peaks_from_density(density, prob_threshold, radius_px)
            if len(peaks) == 0:
                continue
            x0, y0 = case.patch_xy[idx]
            level0 = peaks.copy()
            level0[:, 0] = x0 + peaks[:, 0] * case.downsample
            level0[:, 1] = y0 + peaks[:, 1] * case.downsample
            detections.append(level0)

    if not detections:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(detections, axis=0).astype(np.float32)
