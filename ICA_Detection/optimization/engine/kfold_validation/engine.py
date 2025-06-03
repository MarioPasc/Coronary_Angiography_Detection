"""
engine.py
=========

Run a **k-fold cross-validation** using the *best* hyper-parameter trials found by
`fetch_best_trial.py`.

The script:
1.  Calls the HPO scanner to get a DataFrame of winning `args.yaml` paths.
2.  Checks which optimisers are missing per model_variant and warns.
3.  Spawns one worker *per fold* (GPU) that sequentially trains **all**
    <model_variant, optimiser> combinations for that fold.
4.  Each training run writes its usual Ultralytics `results.csv` into:

        <model_variant>/<optimizer>/fold_<N>/

5.  Collects basic metrics into a CSV summary.

Example
-------
python -m ICA_Detection.optimization.engine.validation.engine \
       --base-dir   /media/.../cadica \
       --kfold-dir  /home/.../k_folds \
       --gpu-ids    0,1,2 \
       --out        kfold_summary.csv \
       --verbose
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from ICA_Detection.optimization import LOGGER
from ICA_Detection.optimization.engine.trainers._base import KFoldTrainer
from ICA_Detection.optimization.engine.trainers.dca_yolov8 import DCAYOLOv8Trainer
from ICA_Detection.optimization.engine.trainers.ultralytics import UltralyticsTrainer
from ICA_Detection.optimization.analyze import fetch_best_trial as fbt
# --------------------------------------------------------------------------- #
#  Bridge to fetch_best_trial                                                 #
# --------------------------------------------------------------------------- #
def _collect_best_trials(
    base_dir: Path,
    remove_weights: bool,
) -> pd.DataFrame:
    """
    Import `fetch_best_trial` as a module and reuse its internal helpers to
    harvest the best rows without writing anything to disk.
    """
    results: List[Dict] = []

    for model_dir in fbt._discover_model_variants(base_dir):  # type: ignore[attr-defined]
        missing: List[str] = []
        for optim in fbt._EXPECTED_OPTIMIZERS:                # type: ignore[attr-defined]
            if not (model_dir / optim).exists():
                missing.append(optim)
                continue
            try:
                row = fbt._process_optimizer(model_dir, optim, remove_weights)  # type: ignore[attr-defined]
                results.append(row.__dict__)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("fetch_best_trial failed on %s/%s: %s", model_dir.name, optim, exc)

        if missing:
            LOGGER.warning("Model %s missing optimisers: %s", model_dir.name, ", ".join(missing))

    if not results:
        LOGGER.error("No best trials discovered – aborting.")
        sys.exit(1)

    return pd.DataFrame(results)


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _parse_gpu_ids(text: str) -> List[int]:
    try:
        return [int(tok) for tok in text.split(",") if tok.strip() != ""]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid GPU list '{text}'") from exc


def _prepare_args(original_yaml: Path, fold_yaml: Path, gpu_id: int,
                  project_root: Path, run_name: str) -> Dict:
    """Load the winner args and inject fold-specific overrides."""
    with open(original_yaml, "r", encoding="utf-8") as fh:
        args = yaml.safe_load(fh)

    args.update(
        data=str(fold_yaml),
        device=f"cuda:{gpu_id}",
        project=str(project_root),
        name=run_name,
        val=True,          # just to be explicit
    )
    # Let Ultralytics rebuild its own directory
    args.pop("save_dir", None)
    return args


def _worker(
    fold_idx: int,
    gpu_id: int,
    fold_yaml: Path,
    tasks: List[Dict],
    metrics_q: mp.Queue,
) -> None:
    """
    Sequentially execute *tasks* (dicts with keys: row, out_root) on one GPU.
    Push a metrics dict back through *metrics_q* after each run.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    LOGGER.info("[Worker-GPU%d] Starting fold_%d with %d jobs.", gpu_id, fold_idx, len(tasks))

    for task in tasks:
        row = task["row"]
        out_root: Path = task["out_root"]

        run_name = f"fold_{fold_idx}"
        project_root = out_root  # <model_variant>/<optimizer>

        args = _prepare_args(
            original_yaml=Path(row["args_path"]),
            fold_yaml=fold_yaml,
            gpu_id=gpu_id,
            project_root=project_root,
            run_name=run_name,
        )

        TrainerCls = (
            DCAYOLOv8Trainer.MODEL_CLS
            if row["model_variant"].startswith("dca_")
            else UltralyticsTrainer.MODEL_CLS
        )
        kft = KFoldTrainer(TrainerCls, args)
        metrics = kft.train()
        metrics_q.put(
            {
                **metrics,
                "model_variant": row["model_variant"],
                "optimizer": row["optimizer"],
                "fold": fold_idx,
                "args_path": row["args_path"],
            }
        )

    LOGGER.info("[Worker-GPU%d] Fold_%d done.", gpu_id, fold_idx)


# --------------------------------------------------------------------------- #
#  CLI & main                                                                 #
# --------------------------------------------------------------------------- #
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="k-fold validation engine")
    p.add_argument("--base-dir", type=Path, required=True, help="cadica root used in optimisation")
    p.add_argument("--kfold-dir", type=Path, required=True, help="Directory containing fold_0/…")
    p.add_argument("--gpu-ids", type=_parse_gpu_ids, required=True, help="Comma list, e.g. 0,1,2")
    p.add_argument("--remove-weights", action="store_true", help="Forwarded to fetch_best_trial")
    p.add_argument("--out", type=Path, default=Path("kfold_summary.csv"))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)-8s %(message)s")


def main() -> None:
    args = _parse_cli()
    _setup_logging(args.verbose)

    df_best = _collect_best_trials(args.base_dir, args.remove_weights)

    folds = sorted(args.kfold_dir.glob("fold_*"))
    if len(folds) == 0:
        LOGGER.error("No fold_N directories found under %s", args.kfold_dir)
        sys.exit(1)
    if len(folds) != len(args.gpu_ids):
        LOGGER.error("Need exactly one GPU id per fold (got %d folds, %d GPUs)", len(folds), len(args.gpu_ids))
        sys.exit(1)

    # Assemble task list per fold / GPU
    tasks_per_fold: Dict[int, List[Dict]] = {i: [] for i in range(len(folds))}
    for _, row in df_best.iterrows():
        model_dir = args.base_dir / row["model_variant"]
        out_root = model_dir / row["optimizer"]           # matches requested structure
        for fold_idx, fold_dir in enumerate(folds):
            tasks_per_fold[fold_idx].append(
                dict(row=row.to_dict(), out_root=out_root, fold_dir=fold_dir)
            )

    # Launch workers
    mp.set_start_method("spawn", force=True)
    metrics_q: mp.Queue = mp.Queue()
    workers: List[mp.Process] = []

    for fold_idx, (gpu_id, fold_dir) in enumerate(zip(args.gpu_ids, folds)):
        w_tasks = tasks_per_fold[fold_idx]
        p = mp.Process(
            target=_worker,
            args=(
                fold_idx,
                gpu_id,
                fold_dir / f"fold_{fold_idx}.yaml",
                w_tasks,
                metrics_q,
            ),
            daemon=False,
        )
        p.start()
        workers.append(p)

    # Collect metrics
    all_metrics: List[Dict] = []
    finished = 0
    while finished < len(workers):
        msg = metrics_q.get()
        if msg == "__worker_done__":
            finished += 1
        else:
            all_metrics.append(msg)

    for p in workers:
        p.join()

    # Write summary CSV
    pd.DataFrame(all_metrics).to_csv(args.out, index=False)
    LOGGER.info("k-fold summary stored at %s", args.out)


if __name__ == "__main__":
    main()
