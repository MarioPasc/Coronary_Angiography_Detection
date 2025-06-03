"""
fetch_best_trial.py
===================

Locate the best hyper-parameter optimisation *trial* for every combination of
model-variant (yolov8{s,m,l}/dca_yolov8{s,m,l}) and optimiser
(cmaes, gpsampler, random, tpe, ultralytics_es) inside a
CADiCa-style results tree.

For *Optuna* (.db) studies we pick **study.best_trial** (direction = MAXIMIZE).
For *Ultralytics ES* runs we pick the **trainXX** folder whose `results.csv`
contains the epoch with the highest F1-score (peak F1).

A summary CSV is produced with, among others, a direct path to the *args.yaml*
of the winning trial/run – so we can fully reproduce the training settings.

---------------------------------------------------------------------------
Command-line
---------------------------------------------------------------------------
    python -m ICA_Detection.optimization.analyze.fetch_best_trial \
           --base-dir /path/to/cadica \
           --out       best_trials_summary.csv \
           --verbose
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, cast

import optuna
import pandas as pd
from tabulate import tabulate

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------#
#  Data containers                                                            #
# ---------------------------------------------------------------------------#
@dataclass
class BestTrial:
    model_variant: str
    optimizer: str
    source: str
    score: float
    trial_number: int
    args_path: Path
    precision: Optional[float] = None
    recall: Optional[float] = None
    notes: str = ""


# ---------------------------------------------------------------------------#
#  Optuna helpers                                                             #
# ---------------------------------------------------------------------------#
def _fetch_study_name(db_path: Path) -> str:
    with sqlite3.connect(db_path) as conn:
        names = [row[0] for row in conn.execute("SELECT study_name FROM studies")]
    if not names:
        raise RuntimeError(f"No study found in {db_path}")
    return names[0]


def _best_trial_from_db(db_path: Path) -> Tuple[int, float]:
    study_name = _fetch_study_name(db_path)
    storage_uri = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_uri)
    best = study.best_trial
    return best.number, cast(float, best.values[0])


# ---------------------------------------------------------------------------#
#  Ultralytics ES helpers                                                     #
# ---------------------------------------------------------------------------#
_F1_EPS = 1e-12


def _is_int(text: str) -> bool:
    try:
        int(text)
        return True
    except ValueError:
        return False


def _calc_f1(prec: pd.Series, rec: pd.Series) -> pd.Series:
    return 2 * prec * rec / (prec + rec + _F1_EPS)


def _best_f1_from_results(csv_path: Path) -> Tuple[float, float, float]:
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    f1 = _calc_f1(df["metrics/precision(B)"], df["metrics/recall(B)"])
    idx = int(f1.idxmax())
    return float(f1.iloc[idx]), float(df["metrics/precision(B)"].iloc[idx]), float(
        df["metrics/recall(B)"].iloc[idx]
    )


def _scan_ultralytics_es(folder: Path) -> Tuple[int, float, float, float, Path]:
    best_f1 = -float("inf")
    best_train = -1
    best_prec = best_rec = 0.0
    best_args: Optional[Path] = None

    for sub in folder.iterdir():
        if not (sub.is_dir() and sub.name.startswith("train")):
            continue
        suffix = sub.name[len("train") :]
        if not _is_int(suffix):
            _LOG.debug("Skipping %s (non-numeric suffix)", sub.name)
            continue
        csv_file = sub / "results.csv"
        if not csv_file.is_file():
            _LOG.debug("Skipping %s (no results.csv)", sub)
            continue
        f1, prec, rec = _best_f1_from_results(csv_file)
        _LOG.debug("Run %s  peak-F1=%.4f", sub.name, f1)
        if f1 > best_f1:
            best_f1, best_prec, best_rec = f1, prec, rec
            best_train = int(suffix)
            best_args = sub / "args.yaml"

    if best_train < 0:
        raise RuntimeError(f"No valid results.csv under {folder}")
    assert best_args is not None
    return best_train, best_f1, best_prec, best_rec, best_args


# ---------------------------------------------------------------------------#
#  House-keeping helpers                                                     #
# ---------------------------------------------------------------------------#
def _purge_weights(parent: Path, keep_dir_name: str, prefix: str) -> None:
    for sub in parent.iterdir():
        if not (sub.is_dir() and sub.name.startswith(prefix)):
            continue
        if sub.name == keep_dir_name:
            continue
        tgt = sub / "weights"
        if tgt.exists():
            try:
                shutil.rmtree(tgt)
                _LOG.info("Removed weights: %s", tgt)
            except Exception as exc:
                _LOG.warning("Could not remove %s: %s", tgt, exc)


# ---------------------------------------------------------------------------#
#  Traversal logic                                                            #
# ---------------------------------------------------------------------------#
_EXPECTED_OPTIMIZERS = {"cmaes", "gpsampler", "random", "tpe", "ultralytics_es"}


def _discover_model_variants(base_dir: Path) -> Sequence[Path]:
    return [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(("yolov8", "dca_yolov8"))]


def _process_optimizer(
    model_dir: Path, optimizer_name: str, remove_weights: bool = False
) -> BestTrial:
    opt_dir = model_dir / optimizer_name
    model_variant = model_dir.name

    if optimizer_name == "ultralytics_es":
        train_num, best_f1, prec, rec, args_path = _scan_ultralytics_es(opt_dir)
        if remove_weights:
            _purge_weights(opt_dir, f"train{train_num}", "train")
        return BestTrial(
            model_variant=model_variant,
            optimizer=optimizer_name,
            source="ultralytics_es",
            score=best_f1,
            trial_number=train_num,
            args_path=args_path,
            precision=prec,
            recall=rec,
            notes="peak F1 across epochs",
        )

    db_path = next(opt_dir.glob("*.db"))
    trial_num, best_value = _best_trial_from_db(db_path)
    root_dir = opt_dir / db_path.stem
    args_path = root_dir / f"trial_{trial_num}" / "args.yaml"
    if remove_weights:
        _purge_weights(root_dir, f"trial_{trial_num}", "trial_")
    return BestTrial(
        model_variant=model_variant,
        optimizer=optimizer_name,
        source="optuna",
        score=best_value,
        trial_number=trial_num,
        args_path=args_path,
        notes="Objective_0 (maximise)",
    )


# ---------------------------------------------------------------------------#
#  CLI & main                                                                 #
# ---------------------------------------------------------------------------#
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch best HP-tuning trials")
    p.add_argument("--base-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("best_trials_summary.csv"))
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--remove-weights",
        action="store_true",
        help="Delete all weights/ folders except the one from the best trial/run",
    )
    return p.parse_args()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)-8s %(message)s")


def main() -> None:
    ns = _parse_cli()
    _setup_logging(ns.verbose)
    results: List[BestTrial] = []
    for model_dir in _discover_model_variants(ns.base_dir):
        for optim in _EXPECTED_OPTIMIZERS:
            if not (model_dir / optim).exists():
                continue
            try:
                results.append(_process_optimizer(model_dir, optim, ns.remove_weights))
            except Exception as exc:  # noqa: BLE001
                _LOG.error("Failed on %s/%s: %s", model_dir.name, optim, exc)

    if not results:
        _LOG.error("Nothing to write")
        return

    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(ns.out, index=False)
    print(tabulate(df[["model_variant", "optimizer", "score", "trial_number", "args_path"]], headers="keys", tablefmt="github"))
    _LOG.info("Summary saved to %s", ns.out)


if __name__ == "__main__":
    main()
