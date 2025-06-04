#!/usr/bin/env python
"""
$ python -m ICA_Detection.optimization.analysis.shap.heatmap_cli \
      --artifacts  ./analysis_outputs \
      --out        ./analysis_outputs/plots_heatmap \
      --fmt        pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ICA_Detection.optimization.analyze.shap.visualization import make_strategy_heatmaps

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)


def _parse(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Generate optimiser-specific heat-maps")
    p.add_argument("--artifacts", type=Path, required=True,
                   help="Folder with shap_dataset and SHAP artefacts")
    p.add_argument("--out", type=Path, required=True,
                   help="Directory to save the 10 heat-maps + colour-bar")
    p.add_argument("--fmt", default="pdf",
                   help="Graphic format (default: pdf)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns = _parse(argv)
    make_strategy_heatmaps(ns.artifacts, ns.out, ns.fmt)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
