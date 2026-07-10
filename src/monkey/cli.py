"""Command-line interface for the MONKEY detection package.

Subcommands:
    smoke    synthetic forward pass on CPU (no data, no checkpoint)
    train    leave-one-centre-out training with checkpoint/resume
    detect   run detection on one case, write predicted points
    oof      out-of-fold evaluation (delegates to monkey.evaluate)
    figures  regenerate figures (delegates to the figure scripts)

Heavy modules are imported lazily inside each handler so ``smoke`` stays fast.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MONKEY inflammatory-cell detection",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("smoke", help="Synthetic forward pass on CPU (no data)")

    p_train = sub.add_parser("train", help="Leave-one-centre-out training")
    p_train.add_argument("--data-dir", default=None,
                         help="Directory of per-case HDF5 (default: MONKEY_DATA_DIR)")
    p_train.add_argument("--ckpt-dir", default=None,
                         help="Where to save fold_{i}_{tag}.pt")
    p_train.add_argument("--device", default=None, choices=["cpu", "cuda"])
    p_train.add_argument("--epochs", type=int, default=None,
                         help="Override the number of epochs")
    p_train.add_argument("--folds", default=None,
                         help="Comma-separated 1-indexed folds (default: all)")

    p_detect = sub.add_parser("detect", help="Detect cells in one case")
    p_detect.add_argument("--case", required=True, help="Path to a case .h5")
    p_detect.add_argument("--ckpt-dir", default=None)
    p_detect.add_argument("--fold", type=int, default=1,
                          help="Fold checkpoint to use (default 1)")
    p_detect.add_argument("--device", default=None, choices=["cpu", "cuda"])
    p_detect.add_argument("--prob-threshold", type=float, default=0.0)
    p_detect.add_argument("--out", default=None,
                          help="Output CSV path (default: results/<case>_points.csv)")

    p_oof = sub.add_parser("oof", help="Out-of-fold evaluation")
    p_oof.add_argument("--data-dir", default=None)
    p_oof.add_argument("--ckpt-dir", default=None)
    p_oof.add_argument("--device", default=None, choices=["cpu", "cuda"])
    p_oof.add_argument("--out", default=None,
                       help="Output metrics.json (default: results/metrics.json)")
    p_oof.add_argument("--max-cases", type=int, default=None)

    p_fig = sub.add_parser("figures", help="Regenerate figures")
    p_fig.add_argument("--ckpt-dir", default=None)
    p_fig.add_argument("--data-dir", default=None)
    p_fig.add_argument("--out", default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "smoke":
        _run_smoke()
    elif args.command == "train":
        _run_train(args)
    elif args.command == "detect":
        _run_detect(args)
    elif args.command == "oof":
        _run_oof(args)
    elif args.command == "figures":
        _run_figures(args)
    else:
        parser.error(f"Unknown command: {args.command}")


def _resolve_device(name: str | None):
    import torch

    if name == "cuda" or (name is None and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _run_smoke() -> None:
    import torch

    from .config import N_CLASSES, TILE_SIZE
    from .model import UNetDensity, count_params

    torch.manual_seed(0)
    x = torch.randn(1, 3, TILE_SIZE, TILE_SIZE)
    print("smoke ok")
    print(f"  input:  {tuple(x.shape)}")
    for head in ("conv", "kan"):
        model = UNetDensity(in_ch=3, n_classes=N_CLASSES, head=head).eval()
        with torch.no_grad():
            out = model(x)
        counts = count_params(model)
        assert out.shape == (1, N_CLASSES, TILE_SIZE, TILE_SIZE), out.shape
        assert torch.isfinite(out).all(), "non-finite output"
        print(f"  head={head}: output {tuple(out.shape)}  "
              f"params total {counts['total']:,}  head {counts['head']:,}  "
              f"kan {counts['kan']:,}")


def _run_train(args) -> None:
    from .config import NUM_EPOCHS
    from .train import train

    device = _resolve_device(args.device)
    folds = [int(x) for x in args.folds.split(",")] if args.folds else None
    train(
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        device=device,
        num_epochs=args.epochs or NUM_EPOCHS,
        folds=folds,
    )


def _run_detect(args) -> None:
    import csv
    from pathlib import Path

    from .checkpoint import load_fold_model
    from .data import MonkeyCase
    from .detect import detect_case

    device = _resolve_device(args.device)
    case = MonkeyCase(args.case)
    model = load_fold_model(args.fold, ckpt_dir=args.ckpt_dir, device=device)
    points = detect_case(model, case, device, prob_threshold=args.prob_threshold)

    out = Path(args.out) if args.out else (
        Path("results") / f"{case.case}_points.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x0", "y0", "score", "class"])
        for x0, y0, score, cls in points:
            writer.writerow([f"{x0:.2f}", f"{y0:.2f}", f"{score:.4f}", int(cls)])
    print(f"{case.case}: {len(points)} detections")
    print(f"points written: {out}")


def _run_oof(args) -> None:
    from .evaluate import evaluate_oof

    device = _resolve_device(args.device)
    evaluate_oof(
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        device=device,
        out_path=args.out,
        max_cases=args.max_cases,
    )


def _run_figures(args) -> None:
    import os
    import runpy
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
    make_figures = scripts_dir / "make_figures.py"
    if not make_figures.exists():
        print(f"figures script not found: {make_figures}. "
              f"Expected scripts/make_figures.py.")
        return
    if args.ckpt_dir:
        os.environ["MONKEY_CKPT_DIR"] = args.ckpt_dir
    if args.data_dir:
        os.environ["MONKEY_DATA_DIR"] = args.data_dir
    if args.out:
        os.environ["MONKEY_FIGURES_DIR"] = args.out
    runpy.run_path(str(make_figures), run_name="__main__")


if __name__ == "__main__":
    main()
