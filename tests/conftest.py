"""Shared pytest fixtures (offline, deterministic, CPU-only, no real data)."""

from pathlib import Path

import h5py
import numpy as np
import pytest


def _write_case(path: Path, case: str, center: str, n_tiles: int = 2,
                tile_size: int = 64, points=None,
                base_mpp: float = 0.242, downsample: float = 2.0) -> Path:
    rng = np.random.RandomState(abs(hash(case)) % (2 ** 31))
    patches = rng.randint(0, 256, size=(n_tiles, tile_size, tile_size, 3),
                          dtype=np.uint8)
    roi = np.ones((n_tiles, tile_size, tile_size), dtype=np.uint8)
    patch_xy = np.array([[i * 1000, i * 2000] for i in range(n_tiles)],
                        dtype=np.int32)
    pts = np.asarray(points if points is not None else [], dtype=np.float32)
    if pts.size == 0:
        pts = np.zeros((0, 4), dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("patches", data=patches)
        f.create_dataset("roi_mask", data=roi)
        f.create_dataset("patch_xy", data=patch_xy)
        f.create_dataset("points", data=pts)
        f.attrs.update(case=case, center=center, level=1,
                       downsample=downsample, base_mpp=base_mpp,
                       tile_size=tile_size)
    return path


@pytest.fixture
def make_case(tmp_path):
    """Factory writing a small synthetic per-case HDF5 to a temp directory."""
    def _factory(case="A_01", center="A", **kw):
        return _write_case(tmp_path / f"{case}.h5", case, center, **kw)
    return _factory


@pytest.fixture
def case_index():
    """A synthetic ``(case, center, path)`` index over four centres."""
    rows = []
    for center in ("A", "B", "C", "D"):
        for k in range(3):
            case = f"{center}_{k:02d}"
            rows.append((case, center, Path(f"/tmp/{case}.h5")))
    return rows
