"""KAN-head interpretability figures.

The density regressor's optional KAN decoder head is a single cross-channel
``KANLinear`` with one learnable B-spline per (input, output) channel edge.
This script reads a trained KAN-head checkpoint and draws the learned per-edge
spline activations, a per-edge nonlinearity summary (how far each edge is from
a straight line), and the spline-scaler magnitudes. Needs a KAN checkpoint, so
it runs on Colab (imports torch).
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from figstyle import apply_style, results_dir, save  # noqa: E402
from monkey.checkpoint import load_fold_model  # noqa: E402

NEAR_LINEAR_THRESH = 0.10
N_EDGES_PLOT = 8
EDGE_GRID_N = 200


def _kan_site(model):
    head = getattr(model, "head", None)
    kan = getattr(head, "kan", None)
    if kan is None:
        raise RuntimeError(
            "loaded checkpoint has no KAN head; train with MONKEY_HEAD=kan "
            "and point MONKEY_KAN_TAG at that run.")
    return kan


def edge_curves(kan) -> tuple[np.ndarray, np.ndarray]:
    """(xs, curves[out, in, N]) sampled spline responses per edge."""
    xs, curves = kan.edge_functions(n_points=EDGE_GRID_N)
    return xs.cpu().numpy(), curves.cpu().numpy()


def nonlinearity_scores(curves: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Per-edge nonlinearity in [0, 1]: residual from best linear fit / total."""
    out_f, in_f, n = curves.shape
    flat = curves.reshape(-1, n)
    design = np.stack([xs, np.ones_like(xs)], axis=1)
    coef, *_ = np.linalg.lstsq(design, flat.T, rcond=None)
    fit = (design @ coef).T
    resid = flat - fit
    tot = flat - flat.mean(axis=1, keepdims=True)
    nl = resid.std(axis=1) / (tot.std(axis=1) + 1e-8)
    return nl.reshape(out_f, in_f)


def fig_edges(kan, out_dir: Path):
    apply_style()
    xs, curves = edge_curves(kan)
    nl = nonlinearity_scores(curves, xs)
    order = np.argsort(nl.ravel())[::-1][:N_EDGES_PLOT]
    in_f = curves.shape[1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.suptitle("KAN head: learned per-edge spline activations phi(x)\n"
                 f"(top-{N_EDGES_PLOT} most nonlinear edges)", fontsize=12)
    for e in order:
        o, i = e // in_f, e % in_f
        ax.plot(xs, curves[o, i], lw=1.4, alpha=0.85,
                label=f"({i}->{o}) nl={nl[o, i]:.2f}")
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.axvline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("pre-activation x (post InstanceNorm, spline domain)")
    ax.set_ylabel("spline output phi(x)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out_dir, "kan_edge_functions.png")
    return nl


def fig_nonlinearity(kan, nl: np.ndarray, out_dir: Path):
    apply_style()
    fig, (axh, axb) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("KAN head: how nonlinear are the edges?", fontsize=13)
    im = axh.imshow(nl, aspect="auto", cmap="magma", vmin=0,
                    vmax=max(0.3, float(np.percentile(nl, 99))))
    axh.set_xlabel("input channel i")
    axh.set_ylabel("output channel o")
    frac_lin = float((nl < NEAR_LINEAR_THRESH).mean())
    axh.set_title(f"per-edge nonlinearity\n{100 * frac_lin:.0f}% near-linear "
                  f"(nl<{NEAR_LINEAR_THRESH:g}), median={np.median(nl):.2f}",
                  fontsize=10)
    fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04)

    axb.hist(nl.ravel(), bins=40, color="steelblue", alpha=0.85)
    axb.axvline(NEAR_LINEAR_THRESH, color="k", ls="--", alpha=0.6,
                label=f"near-linear cutoff ({NEAR_LINEAR_THRESH:g})")
    axb.set_xlabel("per-edge nonlinearity score")
    axb.set_ylabel("edge count")
    axb.set_title("distribution of edge nonlinearity")
    axb.legend(fontsize=8)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out_dir, "kan_nonlinearity.png")
    return {"pct_near_linear": 100 * frac_lin,
            "median_nl": float(np.median(nl)), "max_nl": float(nl.max()),
            "n_edges": int(nl.size)}


def fig_scaler(kan, out_dir: Path):
    apply_style()
    mag = kan.spline_scaler.detach().abs().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.suptitle("KAN head: |spline_scaler| per edge (spline-path gain)",
                 fontsize=12)
    im = ax.imshow(mag, aspect="auto", cmap="viridis")
    ax.set_xlabel("input channel i")
    ax.set_ylabel("output channel o")
    ax.set_title(f"mean |scaler| = {mag.mean():.3f}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out_dir, "kan_spline_scaler.png")


def main():
    import json

    fold = int(os.environ.get("MONKEY_KAN_FOLD", "1"))
    tag = os.environ.get("MONKEY_KAN_TAG", "kan_aug")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = results_dir() / "interpretability"
    print(f"KAN interpretability: fold {fold}, tag '{tag}', device {device}")
    try:
        model = load_fold_model(fold, tag=tag, head="kan", device=device)
    except FileNotFoundError as e:
        print(f"  {e}")
        print("  train a KAN-head run (MONKEY_HEAD=kan) to enable this figure.")
        return
    kan = _kan_site(model)
    nl = fig_edges(kan, out_dir)
    summary = fig_nonlinearity(kan, nl, out_dir)
    fig_scaler(kan, out_dir)
    with open(out_dir / "kan_interpretability_summary.json", "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"kan_interpretability_summary.json ({summary['n_edges']} edges)")


if __name__ == "__main__":
    main()
