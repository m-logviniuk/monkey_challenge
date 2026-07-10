"""Density-target construction and points -> density -> peak round-trip."""

import numpy as np

from monkey.detect import peaks_from_density
from monkey.targets import points_to_density


def test_density_peaks_and_roi_mask():
    tile = 128
    roi = np.ones((tile, tile), dtype=np.uint8)
    points = np.array([[30, 40, 0], [90, 100, 1]], dtype=np.float32)
    density = points_to_density(points, roi, sigma_px=3.0, tile_size=tile)
    assert density.shape == (2, tile, tile)
    assert abs(density[0, 40, 30] - 1.0) < 1e-3
    assert abs(density[1, 100, 90] - 1.0) < 1e-3
    # Class 0 point does not appear on the class 1 channel.
    assert density[1, 40, 30] < 1e-3


def test_roi_mask_zeroes_outside():
    tile = 64
    roi = np.zeros((tile, tile), dtype=np.uint8)
    roi[:32] = 1
    points = np.array([[20, 48, 0]], dtype=np.float32)  # y=48 is outside ROI
    density = points_to_density(points, roi, sigma_px=2.0, tile_size=tile)
    assert density[0, 48, 20] == 0.0


def test_points_density_peak_round_trip():
    tile = 128
    roi = np.ones((tile, tile), dtype=np.uint8)
    points = np.array([[30, 40, 0], [90, 100, 1]], dtype=np.float32)
    density = points_to_density(points, roi, sigma_px=3.0, tile_size=tile)
    peaks = peaks_from_density(density, prob_threshold=0.5,
                               radius_px={0: 8.0, 1: 8.0})
    recovered = {(int(round(x)), int(round(y)), int(cls))
                 for x, y, _, cls in peaks}
    assert (30, 40, 0) in recovered
    assert (90, 100, 1) in recovered
    assert len(peaks) == 2
