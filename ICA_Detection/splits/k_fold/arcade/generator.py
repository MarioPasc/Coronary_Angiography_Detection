"""
Create YOLO-style datasets from an ARCADE K-fold JSON.

Every fold F gets:
    <output_root>/fold_F/
        dataset.yaml
        images/{train,val}/  (symlinks)
        labels/{train,val}/  (symlinks)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

from ICA_Detection.splits.k_fold import LOGGER

# ---------------------------------------------------------------------- #
def construct_datasets(split_json: Path, output_root: Path) -> None:
    """Symlink-based dataset materialisation."""
    folds: Dict[str, dict] = json.loads(split_json.read_text())

    LOGGER.info("Building datasets at %s", output_root.resolve())
    for name, splits in folds.items():
        _build_one_fold(name, splits, output_root)
    LOGGER.info("✅  Finished constructing %d fold datasets.", len(folds))


def _build_one_fold(fold_name: str, splits: dict, root: Path) -> None:
    fold_dir = root / fold_name
    img_dir = fold_dir / "images"
    lbl_dir = fold_dir / "labels"

    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir()

    for split, info in splits.items():                       # train / val
        for kind in ("images", "annotations"):
            src = info["files"][kind]
            tgt_base = (img_dir if kind == "images" else lbl_dir) / split
            tgt_base.mkdir(parents=True, exist_ok=True)

            for _, src_path in src.items():
                dst = tgt_base / Path(src_path).name
                try:
                    dst.symlink_to(src_path)
                except FileExistsError:
                    pass

    # dataset.yaml
    yaml_lines = [
        f"path: {fold_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
        "  0: stenosis",
    ]
    (fold_dir / f"{fold_name}.yaml").write_text("\n".join(yaml_lines))


# ---------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Create YOLO datasets from K-fold JSON.")
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    construct_datasets(args.splits, args.out)


if __name__ == "__main__":
    main()
