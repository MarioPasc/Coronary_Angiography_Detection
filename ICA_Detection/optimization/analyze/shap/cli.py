#!/usr/bin/env python
"""
Command-line wrapper:

$ python -m ICA_Detection.optimization.analysis.shap.cli \
      --cadica-root /path/to/cadica \
      --out         ./analysis_outputs \
      --fmt         parquet         # or csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ICA_Detection.optimization.analyze.shap.data_ingestion import build_dataset
from ICA_Detection.optimization.analyze.shap.model_training import train_and_save_models

logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
)

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SHAP dataset & models")
    p.add_argument("--cadica-root", type=Path, required=True,
                   help="Directory containing yolov8*/ and dca_yolov8*/")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory (dataset, models, shap arrays)")
    p.add_argument("--fmt", choices=("csv", "parquet"), default="parquet",
                   help="Cache file format (default: parquet)")
    p.add_argument("--overwrite", action="store_true",
                   help="Force rebuild even if cache exists")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns = _parse_args(argv)

    if ns.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cache_f = ns.out / f"shap_dataset.{ns.fmt}"
    df = build_dataset(
        cadica_root=ns.cadica_root,
        cache_file=cache_f,
        fmt=ns.fmt,
        overwrite=ns.overwrite,
    )
    train_and_save_models(df, ns.out)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
