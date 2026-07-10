"""Data loading: per-case HDF5 access, ROI-masked density targets, stain and
geometric augmentation, and leave-one-centre-out fold building.

Each case is a PAS whole-slide biopsy tiled to 256x256 patches at ~20x
(``preprocess_nephro.py``). Supervision and scoring happen inside the ROI mask
only; the immunohistochemistry channel is never loaded, so it cannot leak into
the model. Any augmentation is applied to training folds only.
"""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (
    AUG_FLIP_PROB,
    AUG_ROT90_PROB,
    BASE_MPP_DEFAULT,
    CENTERS,
    DOWNSAMPLE_DEFAULT,
    HED_BIAS,
    HED_SIGMA,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MANIFEST_NAME,
    N_CLASSES,
    SIGMA_UM,
    TILE_SIZE,
)
from .targets import points_to_density, sigma_pixels

# Ruifrok-Johnston stain vectors; used only to define a colour space for HED
# jitter, so the exact vectors are not critical to the augmentation.
_RGB_FROM_HED = np.array(
    [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]],
    dtype=np.float32,
)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED).astype(np.float32)

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_STD = np.array(IMAGENET_STD, dtype=np.float32)


def to_model_input(patches: np.ndarray) -> torch.Tensor:
    """Normalise uint8 RGB patches to a model input tensor.

    ``patches`` is ``[N, H, W, 3]`` (or ``[H, W, 3]``) uint8; returns a float
    tensor ``[N, 3, H, W]`` scaled to [0, 1] and ImageNet-normalised.
    """
    arr = np.asarray(patches, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None]
    arr = arr / 255.0
    arr = (arr - _MEAN) / _STD
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(np.ascontiguousarray(arr))


def hed_jitter(patch: np.ndarray, rng: np.random.RandomState,
               sigma: float = HED_SIGMA, bias: float = HED_BIAS) -> np.ndarray:
    """Randomly perturb an RGB patch in HED stain space (train-only).

    Each stain channel is scaled by ``1 + U(-sigma, sigma)`` and shifted by
    ``U(-bias, bias)``, following Tellez et al. (2019). Returns a uint8 patch.
    """
    img = patch.astype(np.float32)
    od = -np.log((img + 1.0) / 256.0)
    stains = od.reshape(-1, 3) @ _HED_FROM_RGB
    alpha = 1.0 + rng.uniform(-sigma, sigma, size=3).astype(np.float32)
    beta = rng.uniform(-bias, bias, size=3).astype(np.float32)
    stains = stains * alpha + beta
    od2 = stains @ _RGB_FROM_HED
    rgb = np.exp(-od2) * 256.0 - 1.0
    rgb = np.clip(rgb, 0.0, 255.0).reshape(img.shape)
    return rgb.astype(np.uint8)


def _geometric(patch: np.ndarray, roi: np.ndarray, density: np.ndarray,
               rng: np.random.RandomState):
    """Apply matched flips and 90-degree rotations to image, ROI, and target."""
    if rng.random() < AUG_FLIP_PROB:
        patch = patch[:, ::-1]
        roi = roi[:, ::-1]
        density = density[:, :, ::-1]
    if rng.random() < AUG_FLIP_PROB:
        patch = patch[::-1, :]
        roi = roi[::-1, :]
        density = density[:, ::-1, :]
    if rng.random() < AUG_ROT90_PROB:
        k = rng.randint(1, 4)
        patch = np.rot90(patch, k, axes=(0, 1))
        roi = np.rot90(roi, k, axes=(0, 1))
        density = np.rot90(density, k, axes=(1, 2))
    return (np.ascontiguousarray(patch), np.ascontiguousarray(roi),
            np.ascontiguousarray(density))


def tile_points(points: np.ndarray, tile_idx: int) -> np.ndarray:
    """Rows ``(x_in_patch, y_in_patch, class)`` for one tile of a case.

    ``points`` is the case's ``[M, 4]`` array with columns
    ``(patch_idx, x, y, class)``.
    """
    if points is None or len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    sel = points[points[:, 0].astype(int) == tile_idx]
    if len(sel) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return sel[:, 1:4].astype(np.float32)


class MonkeyCase:
    """Read-only view of one per-case HDF5 file.

    Loads the tile stack, ROI masks, tile origins, and points into memory and
    exposes the geometry attributes (``mpp``, ``downsample``, ``center``).
    """

    def __init__(self, h5_path: str | Path):
        self.path = Path(h5_path)
        with h5py.File(self.path, "r") as f:
            self.patches = f["patches"][:]
            self.roi_mask = f["roi_mask"][:]
            self.patch_xy = f["patch_xy"][:]
            self.points = f["points"][:]
            attrs = dict(f.attrs)
        self.case = str(attrs.get("case", self.path.stem))
        self.center = str(attrs.get("center", "?"))
        self.downsample = float(attrs.get("downsample", DOWNSAMPLE_DEFAULT))
        base_mpp = float(attrs.get("base_mpp", BASE_MPP_DEFAULT))
        self.base_mpp = base_mpp
        self.mpp = base_mpp * self.downsample
        self.tile_size = int(attrs.get("tile_size", TILE_SIZE))

    @property
    def n_tiles(self) -> int:
        return int(self.patches.shape[0])


def read_case_index(data_dir: str | Path) -> list[tuple[str, str, Path]]:
    """List ``(case, center, h5_path)`` for the dataset.

    Uses ``monkey_manifest.csv`` when present (case, center columns); falls
    back to reading the ``center`` attribute from each ``*.h5``.
    """
    data_dir = Path(data_dir)
    manifest = data_dir / MANIFEST_NAME
    index: list[tuple[str, str, Path]] = []
    if manifest.exists():
        with open(manifest, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                case = row["case"]
                path = data_dir / f"{case}.h5"
                if path.exists():
                    index.append((case, row["center"], path))
        if index:
            return sorted(index, key=lambda r: r[0])
    for path in sorted(data_dir.glob("*.h5")):
        with h5py.File(path, "r") as f:
            center = str(f.attrs.get("center", "?"))
            case = str(f.attrs.get("case", path.stem))
        index.append((case, center, path))
    return sorted(index, key=lambda r: r[0])


def leave_one_centre_out_folds(
    case_index: list[tuple[str, str, Path]],
    centers: tuple[str, ...] = CENTERS,
) -> list[dict]:
    """Build leave-one-centre-out folds.

    Returns one fold per centre: ``{"val_center", "train", "val"}`` where
    ``train`` and ``val`` are lists of ``(case, center, path)``. Every case of
    the held-out centre is in ``val`` and never in ``train``, so no case (and
    no centre) appears on both sides.
    """
    folds = []
    for held in centers:
        val = [r for r in case_index if r[1] == held]
        train = [r for r in case_index if r[1] != held]
        if not val:
            continue
        folds.append({"val_center": held, "train": train, "val": val})
    return folds


class MonkeyDataset(Dataset):
    """Tile-level dataset over one or more cases.

    Yields ``(image, roi, target)``: a normalised ``[3, H, W]`` patch, its
    ``[H, W]`` ROI mask (float32), and the ``[n_classes, H, W]`` density
    target. Stain (HED) and geometric augmentation are applied only when
    ``augment`` is True.
    """

    def __init__(self, case_index: list[tuple[str, str, Path]],
                 augment: bool = False, sigma_um: float = SIGMA_UM,
                 seed: int = 42, n_classes: int = N_CLASSES):
        self.paths = [str(r[2]) for r in case_index]
        self.augment = augment
        self.sigma_um = sigma_um
        self.base_seed = seed
        self.n_classes = n_classes
        self.epoch = 0
        self._rng = np.random.RandomState(seed)
        self._files: dict[int, h5py.File] = {}

        self.index: list[tuple[int, int]] = []
        self._mpp: list[float] = []
        self._downsample: list[float] = []
        self._tile_size: list[int] = []
        for ci, path in enumerate(self.paths):
            with h5py.File(path, "r") as f:
                n = int(f["patches"].shape[0])
                ds = float(f.attrs.get("downsample", DOWNSAMPLE_DEFAULT))
                base_mpp = float(f.attrs.get("base_mpp", BASE_MPP_DEFAULT))
                ts = int(f.attrs.get("tile_size", TILE_SIZE))
            self._mpp.append(base_mpp * ds)
            self._downsample.append(ds)
            self._tile_size.append(ts)
            self.index.extend((ci, ti) for ti in range(n))

    def __len__(self) -> int:
        return len(self.index)

    def _file(self, ci: int) -> h5py.File:
        f = self._files.get(ci)
        if f is None:
            f = h5py.File(self.paths[ci], "r")
            self._files[ci] = f
        return f

    def __getitem__(self, idx: int):
        ci, ti = self.index[idx]
        f = self._file(ci)
        patch = f["patches"][ti]
        roi = f["roi_mask"][ti].astype(np.float32)
        pts = tile_points(f["points"][:], ti)

        sigma_px = sigma_pixels(self._mpp[ci], self.sigma_um)
        tile_size = self._tile_size[ci]
        density = points_to_density(pts, roi, sigma_px, tile_size,
                                    n_classes=self.n_classes)

        if self.augment:
            rng = self._rng
            patch = hed_jitter(patch, rng)
            patch, roi, density = _geometric(patch, roi, density, rng)

        image = to_model_input(patch)[0]
        return (image, torch.from_numpy(np.ascontiguousarray(roi)),
                torch.from_numpy(np.ascontiguousarray(density)))

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    ds._rng = np.random.RandomState(
        ds.base_seed * 100003 + worker_id * 9973 + ds.epoch
    )
    ds._files = {}
