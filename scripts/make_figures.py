"""Regenerate every MONKEY figure and bundle them into figures.zip.

Invoked by ``monkey figures``. Each figure group is run independently and a
missing input (no data, no metrics.json, no checkpoint) only skips that group -
the rest still run. EDA, FROC, and domain-generalization need no model; the
detection and KAN figures need the checkpoints and packed cases and so are run
on Colab. After the figures are written, PNGs under ``results/`` and
``results/interpretability/`` are collected into ``results/figures.zip``.
"""

import sys
import traceback
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from figstyle import results_dir  # noqa: E402


def _run(name: str, fn) -> bool:
    print(f"[{name}]")
    try:
        fn()
        return True
    except Exception as exc:  # keep going even if one group fails
        print(f"  skipped ({type(exc).__name__}): {exc}")
        traceback.print_exc()
        return False


def build_zip(out_dir: Path) -> Path:
    """Collect result PNGs into figures.zip (flat archive)."""
    pngs = sorted(out_dir.glob("*.png"))
    pngs += sorted((out_dir / "interpretability").glob("*.png"))
    zip_path = out_dir / "figures.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in pngs:
            arc = p.name if p.parent == out_dir else f"interpretability/{p.name}"
            zf.write(p, arc)
    print(f"figures.zip ({len(pngs)} figures)")
    return zip_path


def main():
    out_dir = results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    import fig_domain_generalization
    import fig_eda
    import fig_froc

    _run("EDA", fig_eda.main)
    _run("FROC / precision-recall / calibration", fig_froc.main)
    _run("domain generalization", fig_domain_generalization.main)

    # Detection and KAN need torch + checkpoints; import lazily so the
    # lightweight figures still run where torch is unavailable.
    try:
        import fig_detection
        _run("detection overlays", fig_detection.main)
    except Exception as exc:
        print(f"[detection overlays]\n  skipped import ({type(exc).__name__}): {exc}")
    try:
        import fig_kan_interpretability
        _run("KAN interpretability", fig_kan_interpretability.main)
    except Exception as exc:
        print(f"[KAN interpretability]\n  skipped import "
              f"({type(exc).__name__}): {exc}")

    build_zip(out_dir)
    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main()
