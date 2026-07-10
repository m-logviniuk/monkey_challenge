"""Leave-one-centre-out training with per-fold checkpoint/resume.

Each fold holds out one centre for validation and trains on the other three.
The loss is a ROI-masked, foreground-weighted BCE on the sigmoid density map,
so tissue outside the annotated ROI carries no gradient. Best-epoch selection
uses the tuple ``(val_froc, -val_loss)``: validation FROC when scoring is
available, breaking ties on the lower masked validation loss.

A real run needs a GPU and the packed HDF5 cases; on CPU it is only practical
for a short sanity check. Output is per-epoch lines inside the loop, a per-fold
``done in`` line, and a final cross-validation summary.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .checkpoint import build_model, fold_ckpt_path, load_checkpoint
from .config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_DIR,
    EARLY_STOP_PATIENCE,
    GRAD_CLIP,
    HEAD,
    LEARNING_RATE,
    LOSS_FG_WEIGHT,
    MIN_LR_RATIO,
    NUM_EPOCHS,
    NUM_WORKERS,
    POLY_POWER,
    PROB_THRESHOLDS,
    SEED,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
    config_tag,
)
from .data import (
    MonkeyCase,
    MonkeyDataset,
    leave_one_centre_out_folds,
    read_case_index,
    worker_init_fn,
)
from .detect import detect_case


def masked_density_loss(logits: torch.Tensor, target: torch.Tensor,
                        roi: torch.Tensor,
                        fg_weight: float = LOSS_FG_WEIGHT) -> torch.Tensor:
    """ROI-masked, foreground-weighted BCE on the sigmoid density map.

    Pixels outside the ROI are excluded; positive density regions are weighted
    up by ``1 + fg_weight * target`` so sparse cell peaks are not swamped by
    the background.
    """
    weight = (1.0 + fg_weight * target) * roi.unsqueeze(1)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * weight).sum() / weight.sum().clamp(min=1.0)


def _lr_lambda(total_steps: int):
    warmup = max(int(round(total_steps * WARMUP_FRACTION)), 1)
    decay = max(total_steps - warmup, 1)

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max((1.0 - (step - warmup) / decay) ** POLY_POWER, MIN_LR_RATIO)

    return fn


def _loader(dataset: MonkeyDataset, batch_size: int, shuffle: bool,
            device: torch.device) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
        drop_last=shuffle, worker_init_fn=worker_init_fn,
        persistent_workers=False,
    )


@torch.no_grad()
def validate_loss(model, loader, device) -> float:
    """Mean ROI-masked validation loss."""
    model.eval()
    total, n = 0.0, 0
    for image, roi, target in loader:
        image = image.to(device, non_blocking=True)
        roi = roi.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(image)
        total += float(masked_density_loss(logits, target, roi).detach())
        n += 1
    return total / max(n, 1)


def _gt_level0(case: MonkeyCase) -> np.ndarray:
    """Ground-truth points as ``[M, 4]`` (x0, y0, score, class) in level-0."""
    pts = case.points
    if pts is None or len(pts) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = np.zeros((len(pts), 4), dtype=np.float32)
    for i, (pi, x, y, cls) in enumerate(pts):
        x0, y0 = case.patch_xy[int(pi)]
        out[i] = (x0 + x * case.downsample, y0 + y * case.downsample, 1.0, cls)
    return out


def _roi_area_mm2(case: MonkeyCase) -> float:
    px = float(case.roi_mask.sum())
    return px * (case.mpp / 1000.0) ** 2


class _FrocValidator:
    """Optional validation FROC for best-epoch selection.

    Probes ``monkey.froc.froc_score`` once; if scoring is unavailable the
    validator disables itself and training selects on validation loss alone.
    """

    def __init__(self):
        self.enabled = True

    def score(self, model, val_cases, device, prob_threshold: float) -> float:
        if not self.enabled:
            return 0.0
        try:
            from .froc import froc_score
            scores = []
            for _, _, path in val_cases:
                case = MonkeyCase(path)
                pred = detect_case(model, case, device,
                                   prob_threshold=prob_threshold)
                gt = _gt_level0(case)
                res = froc_score(pred, gt, _roi_area_mm2(case), case.base_mpp,
                                 class_name="inflammatory")
                scores.append(float(res["froc"]))
            return float(np.mean(scores)) if scores else 0.0
        except NotImplementedError:
            self.enabled = False
            return 0.0
        except Exception:
            self.enabled = False
            return 0.0


def train_fold(fold: dict, fold_i: int, device: torch.device,
               ckpt_dir: Path, tag: str, num_epochs: int = NUM_EPOCHS,
               seed: int = SEED, verbose: bool = True) -> dict:
    """Train one LOCO fold with checkpoint/resume; return its best metrics."""
    tr_ds = MonkeyDataset(fold["train"], augment=True, seed=seed + fold_i)
    va_ds = MonkeyDataset(fold["val"], augment=False, seed=seed)
    if len(tr_ds) == 0 or len(va_ds) == 0:
        raise RuntimeError(
            f"fold {fold_i} (centre {fold['val_center']}): empty dataset "
            f"(train={len(tr_ds)}, val={len(va_ds)})."
        )
    tr_dl = _loader(tr_ds, BATCH_SIZE, shuffle=True, device=device)
    va_dl = _loader(va_ds, BATCH_SIZE, shuffle=False, device=device)

    model = build_model(head=HEAD, device=device)
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                      weight_decay=WEIGHT_DECAY)
    total_steps = max(len(tr_dl), 1) * num_epochs
    scheduler = optim.lr_scheduler.LambdaLR(opt, _lr_lambda(total_steps))

    prob_thr = PROB_THRESHOLDS[0]
    froc_val = _FrocValidator()
    path = fold_ckpt_path(ckpt_dir, fold_i, tag)
    start_epoch = 0
    best = {"epoch": 0, "score": -1.0, "val_froc": 0.0, "val_loss": float("inf"),
            "state_dict": None}
    no_improve = 0

    if path.exists():
        ckpt = load_checkpoint(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best = ckpt["best"]
        no_improve = ckpt.get("no_improve", 0)
        if verbose:
            print(f"  fold {fold_i} (centre {fold['val_center']}): "
                  f"resuming from epoch {start_epoch}")

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    for epoch in range(start_epoch, num_epochs):
        tr_ds.epoch = epoch
        model.train()
        batches = tr_dl
        if verbose and tqdm is not None:
            batches = tqdm(tr_dl, desc=f"fold {fold_i} ep {epoch + 1}",
                           leave=False)
        for image, roi, target in batches:
            image = image.to(device, non_blocking=True)
            roi = roi.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(image)
            loss = masked_density_loss(logits, target, roi)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            scheduler.step()

        val_loss = validate_loss(model, va_dl, device)
        val_froc = froc_val.score(model, fold["val"], device, prob_thr)
        score = (val_froc, -val_loss)
        best_key = (best["val_froc"], -best["val_loss"])
        improved = best["state_dict"] is None or score > best_key
        if verbose:
            print(f"  fold {fold_i} ep {epoch + 1}/{num_epochs} "
                  f"val_loss={val_loss:.4f} val_froc={val_froc:.4f}")

        if improved:
            best = {
                "epoch": epoch + 1, "score": val_froc, "val_froc": val_froc,
                "val_loss": val_loss,
                "state_dict": {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()},
            }
            no_improve = 0
        else:
            no_improve += 1

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best": best,
            "no_improve": no_improve,
            "head": HEAD,
            "tag": tag,
            "val_center": fold["val_center"],
        }, path)

        if no_improve >= EARLY_STOP_PATIENCE:
            if verbose:
                print(f"  fold {fold_i}: early stop at epoch {epoch + 1}")
            break

    tr_ds.close()
    va_ds.close()
    if best["state_dict"] is None:
        raise RuntimeError(f"fold {fold_i}: no improving epoch recorded.")
    return {
        "fold": fold_i, "val_center": fold["val_center"],
        "best_epoch": best["epoch"], "val_froc": best["val_froc"],
        "val_loss": best["val_loss"],
    }


def train(data_dir=None, ckpt_dir=None, device=None, num_epochs=NUM_EPOCHS,
          folds=None, seed=SEED, tag=None) -> list[dict]:
    """Run leave-one-centre-out CV training and save per-fold checkpoints.

    ``folds`` is an optional list of 1-indexed fold numbers (default: all
    centres). Each fold is saved as ``fold_{i}_{tag}.pt`` under ``ckpt_dir``.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    ckpt_dir = Path(ckpt_dir) if ckpt_dir else CHECKPOINT_DIR
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = tag if tag is not None else config_tag()

    case_index = read_case_index(data_dir)
    all_folds = leave_one_centre_out_folds(case_index)
    if not all_folds:
        raise RuntimeError(f"no cases found under {data_dir}.")
    fold_ids = folds if folds else list(range(1, len(all_folds) + 1))

    results = []
    for fold_i in fold_ids:
        fold = all_folds[fold_i - 1]
        t0 = time.time()
        res = train_fold(fold, fold_i, device, ckpt_dir, tag,
                         num_epochs=num_epochs, seed=seed)
        res["minutes"] = (time.time() - t0) / 60.0
        results.append(res)
        print(f"fold {fold_i} (centre {res['val_center']}) done in "
              f"{res['minutes']:.1f} min, froc={res['val_froc']:.3f}, "
              f"val_loss={res['val_loss']:.4f}, epoch {res['best_epoch']}")
        print()

    _print_cv_summary(results, ckpt_dir, tag)
    return results


def _print_cv_summary(results: list[dict], ckpt_dir: Path, tag: str) -> None:
    total_min = sum(r["minutes"] for r in results)
    print("cross-validation complete")
    print(f"total training time (sum across folds): {total_min:.1f} min")
    print()
    for r in results:
        print(f"  fold {r['fold']} (centre {r['val_center']}): "
              f"froc={r['val_froc']:.3f}, val_loss={r['val_loss']:.4f}, "
              f"ep {r['best_epoch']}, {r['minutes']:.1f} min")
    print()
    frocs = np.array([r["val_froc"] for r in results], dtype=float)
    print(f"mean froc: {frocs.mean():.3f} (sd {frocs.std():.3f})")
    print(f"saved {len(results)} fold checkpoints to {ckpt_dir}")
