"""Paths, constants, and runtime flags.

Values are grouped by stage: data contract, density targets, model,
detection, evaluation, and training. Dataset and checkpoint locations are
read from environment variables so the package never hard-codes a
machine-specific path.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = _env_path("MONKEY_RESULTS_DIR", PROJECT_ROOT / "results")

# Directory of per-case HDF5 files monkey/<case>.h5 (set MONKEY_DATA_DIR).
DATA_DIR = _env_path("MONKEY_DATA_DIR", PROJECT_ROOT / "data")
# Directory holding fold_{i}_{tag}.pt checkpoints (set MONKEY_CKPT_DIR).
CHECKPOINT_DIR = _env_path("MONKEY_CKPT_DIR", PROJECT_ROOT / "checkpoints")
MANIFEST_NAME = "monkey_manifest.csv"

SEED = 42

# Data contract (per-case HDF5 from preprocess_nephro.py)
TILE_SIZE = 256
TARGET_MPP = 0.5
# Fallbacks only; the real values are read from each case's HDF5 attrs.
BASE_MPP_DEFAULT = 0.242
DOWNSAMPLE_DEFAULT = 2.0
CENTERS = ("A", "B", "C", "D")
N_CLASSES = 2
CLASS_NAMES = ("lymphocyte", "monocyte")

# Density targets
# Gaussian sigma for a single splatted cell, in microns; converted to pixels
# with each case's patch mpp (base_mpp * downsample).
SIGMA_UM = 3.0
# Peak value of an isolated, ROI-covered cell in the density target.
DENSITY_PEAK = 1.0

# Model
HEAD = _env_str("MONKEY_HEAD", "conv")           # "conv" or "kan"
BASE_CHANNELS = 32
# grid_size + spline_order + 2 == 9 keeps the KAN head parameter-matched to
# the 3x3 convolution it replaces (in_ch == out_ch); see model.build_head.
KAN_GRID_SIZE = 4
KAN_SPLINE_ORDER = 3
# Optional frozen pathology-FM encoder variant (advanced-if-budget only).
USE_FM_ENCODER = _env_flag("MONKEY_USE_FM_ENCODER", default=False)
FM_REPO_ID = _env_str("MONKEY_FM_REPO_ID", "owkin/phikon-v2")

# Detection
# Radius (microns) for peak non-maximum suppression, per class; combined uses
# the inflammatory radius. These mirror the challenge matching tolerances.
MATCH_RADIUS_UM = {"lymphocyte": 4.0, "monocyte": 5.0, "inflammatory": 5.0}
# ImageNet normalisation for the RGB patches.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Evaluation (shared by scoring and figures; kept here so all stages agree)
FP_POINTS = [10, 20, 50, 100, 200, 300]
PROB_THRESHOLDS = [0.4, 0.9]
BOOTSTRAP_RESAMPLES = 2000

# Training
NUM_EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_FRACTION = 0.05
POLY_POWER = 0.9
MIN_LR_RATIO = 1e-2
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 12
NUM_WORKERS = 2
# Weighted MSE emphasis on the positive density regions (weight = 1 + FG * y).
LOSS_FG_WEIGHT = 10.0

# Stain / geometric augmentation (train folds only)
USE_STAIN_AUG = _env_flag("MONKEY_USE_STAIN_AUG", default=True)
HED_SIGMA = 0.05
HED_BIAS = 0.02
AUG_FLIP_PROB = 0.5
AUG_ROT90_PROB = 0.5


def config_tag() -> str:
    """Short identifier for namespacing checkpoints by configuration.

    Includes the decoder head and whether stain augmentation is enabled so a
    with/without-augmentation ablation never overwrites the other's weights.
    """
    aug = "aug" if USE_STAIN_AUG else "noaug"
    fm = "_fm" if USE_FM_ENCODER else ""
    return f"{HEAD}_{aug}{fm}"
