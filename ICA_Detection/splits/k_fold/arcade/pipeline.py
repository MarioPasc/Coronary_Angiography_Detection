"""
High-level wrapper that

1. Builds the *k_fold_splits.json* with :pyfunc:`arcade.splitter.build_kfold_splits`.
2. Materialises symlink-based YOLO datasets with :pyfunc:`arcade.generator.construct_datasets`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ICA_Detection.splits.k_fold import LOGGER
from ICA_Detection.splits.k_fold.arcade.splitter import build_kfold_splits
from ICA_Detection.splits.k_fold.arcade.generator import construct_datasets


# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(meta_json: Path, out_root: Path, k: int = 3, seed: int = 42) -> Path:
    """
    End-to-end execution – returns the path to *k_fold_splits.json*.
    """
    LOGGER.info("=== ARCADE K-fold pipeline started (k=%d, seed=%d) ===", k, seed)

    splits: Dict[str, Any] = build_kfold_splits(meta_json, k=k, seed=seed)
    splits_path = out_root / "k_fold_splits.json"
    out_root.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(splits, indent=2))
    LOGGER.info("Saved split file → %s", splits_path)

    construct_datasets(splits_path, out_root)
    LOGGER.info("=== Pipeline finished ===")
    return splits_path


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Create K-fold splits for ARCADE YOLO.")
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_pipeline(args.meta, args.out, k=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
