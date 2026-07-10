"""Peak detection, NMS, and end-to-end detect_case on a synthetic case."""

import numpy as np
import torch

from monkey.data import MonkeyCase
from monkey.detect import detect_case, peaks_from_density
from monkey.model import UNetDensity


def test_nms_deduplicates_close_peaks():
    density = np.zeros((1, 40, 40), dtype=np.float32)
    density[0, 20, 20] = 1.0
    density[0, 20, 22] = 0.9   # within radius of the stronger peak
    peaks = peaks_from_density(density, prob_threshold=0.5,
                               radius_px={0: 5.0})
    assert len(peaks) == 1
    assert peaks[0, 0] == 20 and peaks[0, 1] == 20


def test_empty_density_returns_empty():
    density = np.zeros((2, 32, 32), dtype=np.float32)
    peaks = peaks_from_density(density, prob_threshold=0.5,
                               radius_px={0: 4.0, 1: 4.0})
    assert peaks.shape == (0, 4)


def test_detect_case_end_to_end(make_case):
    path = make_case(case="A_01", center="A", n_tiles=2, tile_size=64)
    case = MonkeyCase(path)
    torch.manual_seed(0)
    model = UNetDensity(in_ch=3, n_classes=2, c=8, head="conv").eval()
    points = detect_case(model, case, device=torch.device("cpu"),
                         prob_threshold=0.0)
    assert points.ndim == 2 and points.shape[1] == 4
    if len(points) > 0:
        assert set(np.unique(points[:, 3]).astype(int)).issubset({0, 1})
        assert points[:, 0].min() >= case.patch_xy[:, 0].min()
