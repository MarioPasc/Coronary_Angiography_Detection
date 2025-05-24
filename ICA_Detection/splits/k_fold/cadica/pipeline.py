"""
High-level wrapper that runs both phases of the K-fold workflow:

1.  Build the split dictionary (`k_fold_splits.json`).
2.  Construct the YOLO datasets with symbolic links + YAML.

The functions can be imported or executed via CLI:

    from ICA_Detection.k_fold.pipeline import run_pipeline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from ICA_Detection.splits.k_fold import LOGGER
from ICA_Detection.splits.k_fold.cadica.splitter import build_kfold_splits
from ICA_Detection.splits.k_fold.cadica.generator import construct_datasets


# --------------------------------------------------------------------- #
# ---------------------------  PUBLIC API  ---------------------------- #
# --------------------------------------------------------------------- #
def run_pipeline(
    meta_json: Path,
    out_root: Path,
    k: int = 3,
    seed: int = 42,
) -> Path:
    """
    End-to-end execution → returns path to the `k_fold_splits.json` file.

    Parameters
    ----------
    meta_json :
        Original metadata file with per-image annotations.
    out_root :
        Destination directory that will contain `{fold_X}/…` sub-datasets.
    k, seed :
        Standard split parameters.
    """
    LOGGER.info("=== K-fold pipeline started (k=%d, seed=%d) ===", k, seed)

    splits: Dict[str, Any] = build_kfold_splits(meta_json, k=k, seed=seed)

    splits_path = out_root / "k_fold_splits.json"
    out_root.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(splits, indent=2))
    LOGGER.info("Saved split file → %s", splits_path)

    construct_datasets(splits_path, out_root)

    LOGGER.info("=== K-fold pipeline finished ===")
    return splits_path


# --------------------------------------------------------------------- #
# ------------------------------- CLI --------------------------------- #
# --------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Run full K-fold dataset pipeline.")
    ap.add_argument("--meta", required=True, type=Path, help="original metadata JSON")
    ap.add_argument("--out", required=True, type=Path, help="output root directory")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_pipeline(args.meta, args.out, k=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
