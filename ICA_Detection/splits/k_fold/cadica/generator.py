"""
Create YOLO-style data sets from the k-fold split JSON.

Each fold F gets:
    <output_root>/fold_F/
        dataset.yaml
        images/
            train/  symlinks → real images
            test/
        labels/
            train/  symlinks → real YOLO .txt
            test/
Logging uses the project-wide LOGGER exported by ICA_Detection.k_fold.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict

from ICA_Detection.splits.k_fold import LOGGER  


# ---------------------------------------------------------------------- #
# ----------------------------  Core logic  ---------------------------- #
# ---------------------------------------------------------------------- #
def construct_datasets(split_json: Path, output_root: Path) -> None:
    """Build link-based datasets for every fold described in ``split_json``."""
    with split_json.open("r") as fh:
        folds: Dict[str, dict] = json.load(fh)

    LOGGER.info("Building datasets at %s", output_root.resolve())
    for fold_name, splits in folds.items():
        _build_one_fold(fold_name, splits, output_root)

    LOGGER.info("✅  Finished constructing %d fold datasets.", len(folds))


def _build_one_fold(
    fold_name: str,
    splits: dict,
    root: Path,
) -> None:
    fold_dir = root / fold_name
    images_dir = fold_dir / "images"
    labels_dir = fold_dir / "labels"

    # remove old artefacts if they exist
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()

    # ---------- create symlinks for each split ------------------------ #
    for split_name, info in splits.items():  # train / test …
        for kind in ("images", "annotations"):
            src_dict = info["files"][kind]
            target_parent = (images_dir if kind == "images" else labels_dir) / split_name
            target_parent.mkdir(parents=True, exist_ok=True)

            for sample_id, src_path in src_dict.items():
                dst_path = target_parent / Path(src_path).name
                try:
                    dst_path.symlink_to(src_path)
                except FileExistsError:
                    pass  # keep existing link

    LOGGER.info("Fold %-8s | images=%d | labels=%d",
                fold_name,
                sum(1 for _ in (images_dir).rglob("*") if _.is_file()),
                sum(1 for _ in (labels_dir).rglob("*") if _.is_file()),
                )

    # ---------- write dataset YAML ------------------------------------ #
    yaml_lines = [f"path: {fold_dir.resolve()}"]
    for split_name in splits.keys():                          # train, test
        yaml_lines.append(f"{split_name}: images/{split_name}")
    if "val" not in splits:                                   # YOLO needs 'val'
        yaml_lines.append("val: images/val")
    yaml_lines += ["names:", "    0: stenosis"]

    (fold_dir / f"{fold_name}.yaml").write_text("\n".join(yaml_lines))


# ---------------------------------------------------------------------- #
# -------------------------------  CLI  -------------------------------- #
# ---------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Create YOLO datasets from k-fold JSON.")
    ap.add_argument("--splits", required=True, type=Path, help="k_fold_splits.json")
    ap.add_argument("--out", required=True, type=Path, help="output root folder")
    args = ap.parse_args()

    construct_datasets(args.splits, args.out)


if __name__ == "__main__":
    main()
