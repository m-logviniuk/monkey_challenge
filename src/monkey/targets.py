"""Point annotations to per-class Gaussian density targets.

Each annotated cell is splatted as an isotropic Gaussian with unit peak;
overlapping cells are combined by taking the maximum so an isolated cell
always peaks at exactly ``DENSITY_PEAK`` and nearby cells stay separable for
peak detection. The final map is multiplied by the ROI mask so tissue outside
the annotated region carries no supervision signal.
"""

from __future__ import annotations

import numpy as np

from .config import DENSITY_PEAK, N_CLASSES, SIGMA_UM


def sigma_pixels(mpp: float, sigma_um: float = SIGMA_UM) -> float:
    """Gaussian sigma in pixels for a patch at ``mpp`` microns per pixel."""
    return float(sigma_um) / float(mpp)


def points_to_density(points: np.ndarray, roi_mask: np.ndarray,
                      sigma_px: float, tile_size: int,
                      n_classes: int = N_CLASSES,
                      peak: float = DENSITY_PEAK) -> np.ndarray:
    """Build a ``[n_classes, tile, tile]`` density map from tile-local points.

    ``points`` is an array with columns ``(x_in_patch, y_in_patch, class)`` in
    patch pixels; ``roi_mask`` is ``[tile, tile]`` with 1 inside the annotated
    ROI. Returns a ``float32`` map peaked at ``peak`` on each cell and zeroed
    outside the ROI.
    """
    density = np.zeros((n_classes, tile_size, tile_size), dtype=np.float32)
    if points is not None and len(points) > 0:
        radius = max(1, int(round(3.0 * sigma_px)))
        two_sig2 = 2.0 * sigma_px * sigma_px
        for x, y, cls in points:
            ci = int(cls)
            if ci < 0 or ci >= n_classes:
                continue
            xc, yc = float(x), float(y)
            x0 = max(0, int(np.floor(xc)) - radius)
            x1 = min(tile_size, int(np.ceil(xc)) + radius + 1)
            y0 = max(0, int(np.floor(yc)) - radius)
            y1 = min(tile_size, int(np.ceil(yc)) + radius + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            ys = np.arange(y0, y1)[:, None]
            xs = np.arange(x0, x1)[None, :]
            g = peak * np.exp(-(((xs - xc) ** 2 + (ys - yc) ** 2) / two_sig2))
            np.maximum(density[ci, y0:y1, x0:x1], g.astype(np.float32),
                       out=density[ci, y0:y1, x0:x1])
    density *= roi_mask.astype(np.float32)[None]
    return density
