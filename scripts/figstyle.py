"""Shared figure style for the MONKEY result figures.

A clean matplotlib theme: medium figure sizes, simple titles/axes, and PNGs
saved at 150 dpi with deterministic names. Colours are fixed per class and per
centre so every figure is consistent.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASS_COLORS = {"lymphocyte": "#1f77b4", "monocyte": "#d62728",
                "inflammatory": "#2ca02c"}
CENTER_COLORS = {"A": "#4c72b0", "B": "#dd8452", "C": "#55a868", "D": "#c44e52"}
FP_POINTS = [10, 20, 50, 100, 200, 300]


def apply_style() -> None:
    """Apply the shared matplotlib rcParams."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def results_dir() -> Path:
    """Figure output directory.

    Honours ``MONKEY_FIGURES_DIR`` first, then ``MONKEY_RESULTS_DIR`` (the same
    root ``monkey oof`` writes ``metrics.json`` to), then ``./results``.
    """
    root = Path(__file__).resolve().parent.parent
    env = os.environ.get("MONKEY_FIGURES_DIR") or os.environ.get(
        "MONKEY_RESULTS_DIR")
    return Path(env) if env else (root / "results")


def data_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return Path(os.environ.get("MONKEY_DATA_DIR", str(root / "data")))


def metrics_path() -> Path:
    """Path to metrics.json (``MONKEY_METRICS`` or ``results/metrics.json``)."""
    env = os.environ.get("MONKEY_METRICS")
    return Path(env) if env else (results_dir() / "metrics.json")


def save(fig, out_dir: Path, name: str) -> Path:
    """Save ``fig`` to ``out_dir/name`` and report the filename."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"{name}")
    return path
