"""Model construction and per-fold checkpoint I/O.

Checkpoints are namespaced by configuration (``fold_{i}_{tag}.pt``) so a
conv-vs-kan or with-vs-without-augmentation ablation never overwrites another
run's weights. Each checkpoint carries the resume state (model, optimizer,
scheduler, last epoch) and the best-epoch weights selected during training.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import BASE_CHANNELS, CHECKPOINT_DIR, HEAD, N_CLASSES, config_tag
from .model import UNetDensity


def build_model(head: str = HEAD, device: torch.device | str = "cpu",
                in_ch: int = 3, n_classes: int = N_CLASSES,
                c: int = BASE_CHANNELS) -> UNetDensity:
    """Instantiate the density regressor with the requested decoder head."""
    model = UNetDensity(in_ch=in_ch, n_classes=n_classes, c=c, head=head)
    return model.to(device)


def fold_ckpt_path(ckpt_dir: str | Path | None, fold_i: int,
                   tag: str | None = None) -> Path:
    """Path ``fold_{i}_{tag}.pt`` (1-indexed) under ``ckpt_dir``."""
    base = Path(ckpt_dir) if ckpt_dir is not None else CHECKPOINT_DIR
    tag = tag if tag is not None else config_tag()
    return base / f"fold_{fold_i}_{tag}.pt"


def load_checkpoint(path: str | Path,
                    map_location: torch.device | str = "cpu") -> dict:
    """Load a fold checkpoint dict."""
    return torch.load(str(path), weights_only=False, map_location=map_location)


def load_fold_model(fold_i: int, ckpt_dir: str | Path | None = None,
                    tag: str | None = None, head: str = HEAD,
                    device: torch.device | str = "cpu") -> UNetDensity:
    """Build the model and load a fold's best-epoch weights.

    Raises ``FileNotFoundError`` with an actionable message (no silent
    fallback) if the checkpoint is missing.
    """
    path = fold_ckpt_path(ckpt_dir, fold_i, tag)
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {path}. Set MONKEY_CKPT_DIR to a directory "
            f"holding fold_{fold_i}_{tag or config_tag()}.pt, or train first."
        )
    ckpt = load_checkpoint(path, map_location=device)
    state = ckpt.get("best", {}).get("state_dict") or ckpt["model_state_dict"]
    model = build_model(head=ckpt.get("head", head), device=device)
    model.load_state_dict(state)
    model.eval()
    return model
