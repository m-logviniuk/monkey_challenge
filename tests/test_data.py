"""LOCO split, HDF5 indexing, augmentation, and normalisation tests."""

import numpy as np

from monkey.data import (
    MonkeyDataset,
    hed_jitter,
    leave_one_centre_out_folds,
    read_case_index,
    tile_points,
    to_model_input,
)


def test_loco_folds_are_centre_disjoint(case_index):
    folds = leave_one_centre_out_folds(case_index)
    assert len(folds) == 4
    for fold in folds:
        train_centres = {c for _, c, _ in fold["train"]}
        val_centres = {c for _, c, _ in fold["val"]}
        assert val_centres == {fold["val_center"]}
        assert fold["val_center"] not in train_centres
        train_cases = {c for c, _, _ in fold["train"]}
        val_cases = {c for c, _, _ in fold["val"]}
        assert train_cases.isdisjoint(val_cases)


def test_tile_points_filters_by_tile():
    points = np.array([[0, 10, 20, 0], [1, 30, 40, 1], [0, 50, 60, 1]],
                      dtype=np.float32)
    t0 = tile_points(points, 0)
    assert t0.shape == (2, 3)
    assert set(t0[:, 2].astype(int)) == {0, 1}
    assert tile_points(points, 1).shape == (1, 3)
    assert tile_points(points, 5).shape == (0, 3)


def test_to_model_input_shape_and_scale():
    patches = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    x = to_model_input(patches)
    assert tuple(x.shape) == (3, 3, 16, 16)
    assert x.dtype.is_floating_point


def test_hed_jitter_preserves_shape_and_dtype():
    rng = np.random.RandomState(0)
    patch = rng.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    out = hed_jitter(patch, rng)
    assert out.shape == patch.shape
    assert out.dtype == np.uint8


def test_dataset_reads_case_and_targets(make_case):
    path = make_case(case="B_00", center="B", n_tiles=2, tile_size=64,
                     points=[[0, 20, 30, 0], [1, 40, 50, 1]])
    index = read_case_index(path.parent)
    ds = MonkeyDataset(index, augment=True, seed=1)
    assert len(ds) == 2
    image, roi, target = ds[0]
    assert tuple(image.shape) == (3, 64, 64)
    assert tuple(roi.shape) == (64, 64)
    assert tuple(target.shape) == (2, 64, 64)
    ds.close()
