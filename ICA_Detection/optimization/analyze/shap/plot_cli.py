#!/usr/bin/env python
"""
$ python -m ICA_Detection.optimization.analysis.shap.plot_cli \
      --artifacts  ./analysis_outputs \
      --out        ./analysis_outputs/plots \
      --fmt        pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ICA_Detection.optimization.analyze.shap.visualization import make_all_plots

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SHAP importance plots")
    p.add_argument("--artifacts", type=Path, required=True,
                   help="Folder holding global_importance.parquet etc.")
    p.add_argument("--out", type=Path, required=True,
                   help="Destination directory for figures")
    p.add_argument("--fmt", default="pdf",
                   help="Output graphics format (default: pdf)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns = _parse(argv)
    make_all_plots(ns.artifacts, ns.out, ns.fmt)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
