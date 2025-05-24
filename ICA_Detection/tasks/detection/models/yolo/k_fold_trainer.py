"""
Fold-wise YOLO training/validation.

*   One *independent* process per fold.
*   Each process sees **exactly one** GPU (`CUDA_VISIBLE_DEVICES`) so
    `device=0` inside the Ultralytics call is always valid.
*   Metrics/weights land under:

        <project_root>/
            fold_0/
                train/…   val/…
            fold_1/
            …

Usage
-----
python -m ICA_Detection.tasks.detection.models.yolo.engine.k_fold_trainer \\
       --args  args.yaml \\
       --dataset-root  /scratch/kfold_ds \\
       --gpus  0,1,2,3
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import time
import yaml
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, List

from ICA_Detection.external.ultralytics.ultralytics import YOLO

# ------------------------------------------------------------------ #
#  logging (shares format with other ICA_Detection modules)
# ------------------------------------------------------------------ #
LOGGER = logging.getLogger("ICA_Detection.k_fold_trainer")
if not LOGGER.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)


# ------------------------------------------------------------------ #
# ----------------------------- helpers ---------------------------- #
# ------------------------------------------------------------------ #
def _load_args(path: Path) -> Dict[str, Any]:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def _worker(
    fold_dir: Path,
    base_cfg: Dict[str, Any],
    gpu_id: int,
    fold_idx: int,
) -> None:
    """Runs in its own process; trains and validates one fold."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # prepare per-fold training args
    cfg = base_cfg.copy()
    cfg["data"] = str(fold_dir / f"{fold_dir.name}.yaml")
    cfg["device"] = 0  # logical index after masking
    cfg["project"] = str(Path(cfg["project"]) / f"{fold_dir.name}")
    cfg["name"] = cfg.get("name", "run")

    fold_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Fold %d | GPU %d | project=%s", fold_idx, gpu_id, cfg["project"])

    model = YOLO(cfg["model"])
    t0 = time.time()
    model.train(**{k: v for k, v in cfg.items() if k != "model"})
    metrics = model.val()

    # save metrics so caller can aggregate
    metrics_path = Path(cfg["project"]) / "validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    LOGGER.info("Fold %d finished in %.1f min.", fold_idx, (time.time() - t0) / 60)


# ------------------------------------------------------------------ #
# ---------------------------  main API  --------------------------- #
# ------------------------------------------------------------------ #
def train_kfold(
    args_yaml: Path,
    dataset_root: Path,
    gpu_ids: List[int],
) -> None:
    """Launch a process per fold with round-robin GPU assignment."""
    base_cfg = _load_args(args_yaml)

    folds = sorted([p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    if not folds:
        raise RuntimeError(f"No fold directories found under {dataset_root}")

    if not gpu_ids:
        gpu_ids = [-1]  # CPU fallback

    jobs: List[mp.Process] = []
    for i, fold_dir in enumerate(folds):
        gpu = gpu_ids[i % len(gpu_ids)]
        p = mp.Process(
            target=_worker,
            args=(fold_dir, base_cfg, gpu, i),
            daemon=False,
        )
        p.start()
        jobs.append(p)

    # wait
    for p in jobs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Fold process {p.pid} failed (exit={p.exitcode}).")


# ------------------------------------------------------------------ #
# -------------------------------- CLI ---------------------------- #
# ------------------------------------------------------------------ #
def main() -> None:
    ap = argparse.ArgumentParser(description="Train/validate YOLO on every fold.")
    ap.add_argument("--args", required=True, type=Path, help="YAML with training args")
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--gpus", default="", help="comma-separated GPU ids (e.g. 0,1,2)")
    args = ap.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",") if x]  # empty → CPU
    train_kfold(args.args, args.dataset_root, gpu_ids)


if __name__ == "__main__":
    main()
