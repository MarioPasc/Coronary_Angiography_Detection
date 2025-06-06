"""
engine.py
=========

Run k-fold cross-validation on the *best* hyper-parameter trials.

NEW
---
• `run_engine(cfg)`   → call directly from Python with a dict/YAML payload  
• `--config FILE.yaml` CLI  → alternative to the long list of flags  
• workflow unchanged otherwise
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import traceback
from dataclasses import dataclass
from multiprocessing import Manager, Process
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, cast

import pandas as pd
import yaml
from typing_extensions import TypedDict

from ICA_Detection.optimization import LOGGER as _ROOT_LOGGER
from ICA_Detection.optimization.analyze import fetch_best_trial as fbt
from ICA_Detection.optimization.engine.trainers._base import KFoldTrainer
from ICA_Detection.optimization.engine.trainers.dca_yolov8 import DCAYOLOv8Trainer
from ICA_Detection.optimization.engine.trainers.ultralytics import UltralyticsTrainer

LOG_FORMAT = "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Typed config object
# ─────────────────────────────────────────────────────────────────────────────
class EngineCfg(TypedDict, total=False):
    base_dir: Union[str, Path]
    kfold_dir: Union[str, Path]
    gpu_ids: List[int]
    out: Union[str, Path]
    workspace_dir: Union[str, Path]
    remove_weights: bool
    verbose: bool


# ─────────────────────────────────────────────────────────────────────────────
# Data-classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class KFoldTask:
    row: Dict[str, Any]
    out_root: Path


# ════════════════════════════════════════════════════════════════════════════
# Workspace helpers
# ════════════════════════════════════════════════════════════════════════════
def _scan_workspace(workspace: Path) -> pd.DataFrame:
    """
    Return a DataFrame with columns {model_variant, optimizer, args_path}
    for every   <workspace>/<model>/<opt>/best_args.yaml   that exists.
    """
    rows: list[dict[str, Any]] = []
    if not workspace.is_dir():
        return pd.DataFrame()

    for model_dir in workspace.iterdir():
        if not model_dir.is_dir():
            continue
        for opt_dir in model_dir.iterdir():
            if not opt_dir.is_dir():
                continue
            best_yaml = opt_dir / "best_args.yaml"
            if best_yaml.is_file():
                rows.append(
                    dict(
                        model_variant=model_dir.name,
                        optimizer=opt_dir.name,
                        args_path=best_yaml,
                    )
                )
    return pd.DataFrame(rows)


def _materialise_yaml(df_hpo: pd.DataFrame, workspace: Path) -> pd.DataFrame:
    """
    Copy each best-trial YAML from *df_hpo* into the workspace and
    return a new DataFrame whose `args_path` points **inside** the workspace.
    """
    rows: list[dict[str, Any]] = []
    for _, row in df_hpo.iterrows():
        dst_dir = workspace / row["model_variant"] / row["optimizer"]
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst_yaml = dst_dir / "best_args.yaml"
        if not dst_yaml.exists():  # avoid re-copying
            shutil.copy(row["args_path"], dst_yaml)

        rows.append(
            dict(
                model_variant=row["model_variant"],
                optimizer=row["optimizer"],
                args_path=dst_yaml,
            )
        )
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Utility functions
# ════════════════════════════════════════════════════════════════════════════
def _assign_gpus(gpu_ids: List[int], num_folds: int) -> Dict[int, int]:
    """Map fold-index → gpu-index."""
    if len(gpu_ids) == 1:
        return {i: gpu_ids[0] for i in range(num_folds)}
    return {i: gpu_ids[i] for i in range(num_folds)}


def _prepare_args(
    original_yaml: Path,
    fold_yaml: Path,
    gpu_id: int,
    project_root: Path,
    run_name: str,
    save_dir = str
) -> Dict[str, Any]:
    """Load YAML and patch dataset / device / project fields."""
    with open(original_yaml, "r") as f:
        args: Dict[str, Any] = yaml.safe_load(f)

    args["data"] = str(fold_yaml)
    args["device"] = f"cuda:{gpu_id}"
    args["project"] = str(project_root)
    args["name"] = run_name
    args["val"] = True
    args["save_dir"] = save_dir
    return args


# ════════════════════════════════════════════════════════════════════════════
# Worker
# ════════════════════════════════════════════════════════════════════════════
def _worker(
    fold_idx: int,
    gpu_id: int,
    fold_yaml: Path,
    tasks: List[KFoldTask],
    metrics_q,
    gpu_lock=None,
) -> None:
    """Run all tasks for a single fold on one GPU."""
    #prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch, os
    LOGGER.info(">>> CUDA_VISIBLE_DEVICES = ", os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))
    LOGGER.info(">>> torch.cuda.device_count() =", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        LOGGER.info(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")


    for task in tasks:
        row = task.row
        out_root: Path = task.out_root
        run_name = f"fold_{fold_idx}"

        out_root.mkdir(parents=True, exist_ok=True)
        args = _prepare_args(
            original_yaml=Path(row["args_path"]),
            fold_yaml=fold_yaml,
            gpu_id=gpu_id,
            project_root=out_root,
            run_name=run_name,
            save_dir = out_root,
        )

        TrainerCls: Type[Union[DCAYOLOv8Trainer, UltralyticsTrainer]]
        TrainerCls = (
            DCAYOLOv8Trainer if row["model_variant"].startswith("dca_") else UltralyticsTrainer
        )

        kft = KFoldTrainer(TrainerCls.MODEL_CLS, args)

        if gpu_lock:
            gpu_lock.acquire()
        try:
            metrics = kft.train()
            if metrics is None:
                raise RuntimeError("train() returned None")

            rec = {
                **metrics,
                "model_variant": row["model_variant"],
                "optimizer": row["optimizer"],
                "fold": fold_idx,
                "args_path": str(row["args_path"]),
                "status": "success",
            }
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "[GPU%d][fold_%d] %s/%s failed:\n%s",
                gpu_id,
                fold_idx,
                row["model_variant"],
                row["optimizer"],
                traceback.format_exc(),
            )
            rec = dict(
                precision=float("nan"),
                recall=float("nan"),
                f1_score=float("nan"),
                mAP50=float("nan"),
                mAP50_95=float("nan"),
                elapsed=float("nan"),
                model_variant=row["model_variant"],
                optimizer=row["optimizer"],
                fold=fold_idx,
                args_path=str(row["args_path"]),
                status="failed",
                error_msg=str(exc),
            )
        finally:
            if gpu_lock:
                gpu_lock.release()
            metrics_q.put(rec)

    """
    # restore env
    if prev_cuda is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda
    """

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CORE EXECUTION LOGIC extracted into a private function `_run(cfg)`      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def _run(cfg: EngineCfg) -> None:
    """All previous `main()` logic lives here; cfg values are already validated."""
    # ------------------------------------------------------------------#
    # Normalise / fill defaults
    # ------------------------------------------------------------------#
    base_dir: Path = Path(cfg["base_dir"])
    kfold_dir: Path = Path(cfg["kfold_dir"])
    gpu_ids: List[int] = cfg["gpu_ids"]
    out_csv: Path = Path(cfg["out"])
    workspace: Path = Path(cfg.get("workspace_dir", "kfold_results"))
    remove_weights: bool = cfg.get("remove_weights", False)
    verbose: bool = cfg.get("verbose", False)

    if verbose:
        LOGGER.setLevel(logging.DEBUG)
        _ROOT_LOGGER.setLevel(logging.DEBUG)

    # ---- 1. obtain best-trial YAMLs (workspace cache) -----------------
    df_ws = _scan_workspace(workspace)
    if not df_ws.empty:
        LOGGER.info("Using cached best_args.yaml files in %s", workspace)
        df_best = df_ws
    else:
        LOGGER.info("Workspace empty – scanning HPO results in %s …", base_dir)
        results: list[fbt.BestTrial] = []
        for model_dir in fbt._discover_model_variants(base_dir):
            for optim in fbt._EXPECTED_OPTIMIZERS:
                if (model_dir / optim).exists():
                    try:
                        results.append(
                            fbt._process_optimizer(model_dir, optim, remove_weights)
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.error("Failed on %s/%s: %s", model_dir.name, optim, exc)
        if not results:
            LOGGER.error("No best trials found – abort.")
            sys.exit(1)
        df_hpo = pd.DataFrame([r.__dict__ for r in results])
        df_best = _materialise_yaml(df_hpo, workspace)

    if df_best.empty:
        LOGGER.error("Best-trial DataFrame empty – abort.")
        sys.exit(1)

    # ---- 2. folds ------------------------------------------------------
    fold_dirs = sorted(kfold_dir.glob("fold_*"))
    if not fold_dirs:
        LOGGER.error("No fold_* dirs in %s", kfold_dir)
        sys.exit(1)
    num_folds = len(fold_dirs)
    for i, fdir in enumerate(fold_dirs):
        if not (fdir / f"fold_{i}.yaml").is_file():
            LOGGER.error("Missing %s/fold_%d.yaml", fdir, i)
            sys.exit(1)

    # ---- 3. GPU mapping ------------------------------------------------
    if len(gpu_ids) > 1 and len(gpu_ids) != num_folds:
        LOGGER.error("Supply one GPU or #GPUs == #folds (%d)", num_folds)
        sys.exit(1)
    gpu_map = _assign_gpus(gpu_ids, num_folds)

    # ---- 4. build tasks per fold ---------------------------------------
    tasks_per_fold: list[list[KFoldTask]] = [[] for _ in range(num_folds)]
    for _, row in df_best.iterrows():
        meta = dict(
            model_variant=row["model_variant"],
            optimizer=row["optimizer"],
            args_path=row["args_path"],
        )
        out_root = workspace / meta["model_variant"] / meta["optimizer"] / "out"
        for f_idx in range(num_folds):
            tasks_per_fold[f_idx].append(KFoldTask(row=meta, out_root=out_root))

    # ---- 5. spawn workers ---------------------------------------------
    manager = Manager()
    q_metrics = manager.Queue()
    gpu_lock = manager.Lock() if len(gpu_ids) == 1 else None
    procs: list[Process] = []

    for f_idx, fdir in enumerate(fold_dirs):
        p_ = Process(
            target=_worker,
            args=(
                f_idx,
                gpu_map[f_idx],
                fdir / f"fold_{f_idx}.yaml",
                tasks_per_fold[f_idx],
                q_metrics,
                gpu_lock,
            ),
            daemon=False,
        )
        p_.start()
        procs.append(p_)
        LOGGER.debug("Spawned worker fold_%d on GPU %d (PID=%d)", f_idx, gpu_map[f_idx], p_.pid)

    # ---- 6. collect metrics -------------------------------------------
    expected = len(df_best) * num_folds
    collected: list[dict[str, Any]] = []
    LOGGER.info("Waiting for %d metric records …", expected)
    while len(collected) < expected:
        try:
            collected.append(q_metrics.get(timeout=3600))
        except Exception:  # noqa: BLE001
            LOGGER.error("Timeout while waiting for metrics (%d/%d)", len(collected), expected)
            break

    # ---- 7. join / clean ----------------------------------------------
    for p_ in procs:
        p_.join(timeout=120)
        if p_.is_alive():
            LOGGER.warning("Terminating hung worker PID=%d", p_.pid)
            p_.terminate()

    # ---- 8. write CSV --------------------------------------------------
    pd.DataFrame(collected).to_csv(out_csv, index=False)
    LOGGER.info("Summary CSV → %s", out_csv)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def run_engine(cfg: EngineCfg) -> None:
    """Programmatic entry – accepts a dict coming from YAML or elsewhere."""
    # minimal validation for required keys
    for req in ("base_dir", "kfold_dir", "gpu_ids", "out"):
        if req not in cfg:
            raise ValueError(f"Config missing required key: {req}")
    _run(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# CLI 1:  python -m …engine.py [flags]          (unchanged)
# CLI 2:  python -m …engine.py --config file.yaml
# ─────────────────────────────────────────────────────────────────────────────
def cli() -> None:
    parser = argparse.ArgumentParser(description="Run k-fold CV on best trials")
    parser.add_argument("--config", type=Path, help="YAML file with all parameters")
    # flags identical to previous version (still usable for debugging)
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--kfold-dir", type=Path)
    parser.add_argument("--gpu-ids", type=lambda s: [int(x) for x in s.split(",")])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workspace-dir", type=Path, default=Path("kfold_results"))
    parser.add_argument("--remove-weights", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            parsed_yaml = yaml.safe_load(f)
            # Ensure that parsed_yaml is a dictionary. If it's None (empty file)
            # or not a dictionary (e.g. a list or scalar from YAML), default to an empty dict.
            config_dict = parsed_yaml if isinstance(parsed_yaml, dict) else {}
            cfg: EngineCfg = cast(EngineCfg, config_dict)
        run_engine(cfg)
    else:
        # keep old behaviour – build cfg from flags
        run_engine(
            EngineCfg(
                base_dir=args.base_dir,
                kfold_dir=args.kfold_dir,
                gpu_ids=args.gpu_ids,
                out=args.out,
                workspace_dir=args.workspace_dir,
                remove_weights=args.remove_weights,
                verbose=args.verbose,
            )
        )


if __name__ == "__main__":
    cli()