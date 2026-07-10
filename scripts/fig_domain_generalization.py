"""Domain-generalization figures: per-centre FROC and the stain-aug ablation.

The left panel shows the pooled out-of-fold inflammatory FROC per held-out
centre with BCa 95% confidence intervals - the multi-centre generalization
readout. The right panel compares the with- and without-stain-augmentation
runs when a second metrics file is provided via ``MONKEY_METRICS_NOAUG``; the
augmentation ablation is the domain-generalization lever.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402

from figstyle import CENTER_COLORS, apply_style, metrics_path, results_dir, save  # noqa: E402


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _centre_froc(metrics: dict):
    """Per-centre inflammatory FROC + CI arrays, ordered by centre name."""
    per_centre = metrics.get("per_centre", {})
    centres = sorted(per_centre.keys())
    froc, lo, hi = [], [], []
    for c in centres:
        infl = per_centre[c].get("inflammatory", {})
        f = infl.get("froc", 0.0)
        ci = infl.get("froc_ci", [f, f])
        froc.append(f)
        lo.append(f - ci[0])
        hi.append(ci[1] - f)
    return centres, froc, lo, hi


def figure(metrics: dict, metrics_noaug: dict | None, out_dir: Path):
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Domain generalization across centres "
                 "(pooled out-of-fold FROC)", fontsize=13)

    centres, froc, lo, hi = _centre_froc(metrics)
    x = np.arange(len(centres))
    ax1.bar(x, froc, yerr=[lo, hi], capsize=5,
            color=[CENTER_COLORS.get(c, "gray") for c in centres])
    overall = metrics["overall"].get("inflammatory", {}).get("froc")
    if overall is not None:
        ax1.axhline(overall, color="k", ls="--", lw=1,
                    label=f"pooled overall ({overall:.3f})")
        ax1.legend(fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"held-out {c}" for c in centres])
    ax1.set_ylabel("inflammatory FROC")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("per held-out centre (BCa 95% CI)")

    if metrics_noaug is not None:
        c2, froc2, _, _ = _centre_froc(metrics_noaug)
        common = [c for c in centres if c in c2]
        aug_v = [froc[centres.index(c)] for c in common]
        noaug_v = [froc2[c2.index(c)] for c in common]
        xc = np.arange(len(common))
        w = 0.38
        ax2.bar(xc - w / 2, aug_v, w, label="stain-aug", color="#4c72b0")
        ax2.bar(xc + w / 2, noaug_v, w, label="no stain-aug", color="#c44e52")
        ax2.set_xticks(xc)
        ax2.set_xticklabels([f"held-out {c}" for c in common])
        ax2.set_ylabel("inflammatory FROC")
        ax2.set_ylim(0, 1.02)
        ax2.set_title("with vs without stain augmentation")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5,
                 "set MONKEY_METRICS_NOAUG to a no-augmentation\n"
                 "metrics.json to draw the stain-aug ablation",
                 ha="center", va="center", fontsize=10, wrap=True)
        ax2.set_axis_off()

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out_dir, "fig_domain_generalization.png")


def main():
    metrics = _load(metrics_path())
    if metrics is None:
        print(f"metrics.json not found at {metrics_path()}; run `monkey oof`.")
        return
    noaug_env = os.environ.get("MONKEY_METRICS_NOAUG")
    metrics_noaug = _load(Path(noaug_env)) if noaug_env else None
    figure(metrics, metrics_noaug, results_dir())


if __name__ == "__main__":
    main()
