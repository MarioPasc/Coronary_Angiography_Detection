"""
Parse the CADiCA optimisation directories into a tidy trial–level DataFrame.

Public entry-point
------------------
>>> build_dataset(cadica_root=Path("…/cadica"),
...               cache_file=Path("./outputs/shap_dataset.parquet"))
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import optuna
import pandas as pd
import yaml

from .constants import (
    ALL_COLS,
    FAMILY_COLS,
    HP_COLS,
    METRIC_COLS,
    MODEL_SIZES,
    OPT_LIST,
    DatasetFmt,
)

LOGGER = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Small helpers
# ────────────────────────────────────────────────────────────────────────────

_ITER_LINE = re.compile(r"\(\s*([0-9.]+)s\)")

@dataclass(slots=True)
class TrialRow:
    """Internal container for one training run (Optuna trial or Ultralytics folder)."""

    model_family: str
    size: str
    optimiser: str
    trial_id: str
    params: Dict[str, float | int | str | None]
    metrics: Dict[str, float | int | str | None]


# ────────────────────────────────────────────────────────────────────────────
# Public function
# ────────────────────────────────────────────────────────────────────────────

def build_dataset(
    cadica_root: Path,
    cache_file: Path,
    fmt: DatasetFmt = "parquet",
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Return a tidy DataFrame with *one row per trial*, reading from *cache_file*
    when possible.

    Parameters
    ----------
    cadica_root : Path
        Folder that contains `yolov8*/` and `dca_yolov8*/` sub-folders.
    cache_file : Path
        Destination / source of the cached dataset.
    fmt : {'csv','parquet'}
        File format used for caching.
    overwrite : bool
        Ignore cache and rebuild from scratch.

    Notes
    -----
    * All hyper-parameter columns are guaranteed to exist; missing values are
      stored as `NaN`.
    * Execution-time for Ultralytics-ES trials is taken from the comment in
      `best_hyperparameters.yaml` (seconds).
    """
    if cache_file.exists() and not overwrite:
        LOGGER.info("Reloading cached dataset from %s", cache_file)
        return _load_dataframe(cache_file, fmt)

    cadica_root = cadica_root.expanduser().resolve()
    LOGGER.info("Parsing CADiCA tree under %s …", cadica_root)

    rows: List[TrialRow] = []
    for model_dir in sorted(cadica_root.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        family = "dca_yolo" if model_name.startswith("dca_") else "yolo"
        size = model_name[-1]  # 's' / 'm' / 'l'

        for opt_dir in model_dir.iterdir():
            optimiser = opt_dir.name
            if optimiser not in OPT_LIST or not opt_dir.is_dir():
                continue

            if optimiser == "ultralytics_es":
                rows.extend(_collect_ultralytics_rows(opt_dir, family, size))
            else:
                rows.extend(_collect_optuna_rows(opt_dir, family, size, optimiser))

    df = pd.DataFrame([_row_to_dict(r) for r in rows], columns=ALL_COLS)
    LOGGER.info("Built dataset: %d trials × %d columns", len(df), df.shape[1])

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    _save_dataframe(df, cache_file, fmt)
    LOGGER.info("Cached dataset to %s", cache_file)
    return df


# ────────────────────────────────────────────────────────────────────────────
# Internal – dataframe I/O
# ────────────────────────────────────────────────────────────────────────────

def _save_dataframe(df: pd.DataFrame, f: Path, fmt: DatasetFmt) -> None:
    if fmt == "csv":
        df.to_csv(f, index=False)
    else:
        df.to_parquet(f, index=False)


def _load_dataframe(f: Path, fmt: DatasetFmt) -> pd.DataFrame:
    return pd.read_csv(f) if fmt == "csv" else pd.read_parquet(f)


# ────────────────────────────────────────────────────────────────────────────
# Internal – row construction helpers
# ────────────────────────────────────────────────────────────────────────────

def _row_to_dict(r: TrialRow) -> Dict[str, float | int | str | None]:
    base: Dict[str, float | int | str | None] = {
        "model_family": r.model_family,
        "size":         r.size,
        "optimiser":    r.optimiser,
        "trial_id":     r.trial_id,
    }
    base.update({col: r.params.get(col)   for col in HP_COLS})
    base.update({col: r.metrics.get(col)  for col in METRIC_COLS})
    return base


# ---------- Ultralytics-ES --------------------------------------------------

def _collect_ultralytics_rows(opt_dir: Path, family: str, size: str) -> List[TrialRow]:
    rows: List[TrialRow] = []
    ultra_root = _find_ultra_root(opt_dir)
    print(f"Ultralytics root: {ultra_root}")
    if ultra_root is None:
        LOGGER.warning("Cannot locate Ultralytics train folders in %s", opt_dir)
        return rows

    if not ultra_root.exists():
        LOGGER.warning("Missing %s – skipped", ultra_root)
        return rows

    for train_dir in sorted(d for d in ultra_root.iterdir() if d.is_dir() and d.name.startswith("train")):
        print(f"Processing {train_dir}")
        trial_id = train_dir.name
        if trial_id == "train": trial_id = "0"
        args_f   = train_dir / "args.yaml"
        res_f    = train_dir / "results.csv"
        if not (args_f.exists() and res_f.exists()):
            LOGGER.debug("Missing files for %s", train_dir)
            continue

        print(f"Trial ID: {trial_id}, Args: {args_f}, Results: {res_f}")
        params = _filter_hp(yaml.safe_load(args_f.read_text()))

        # ---------------- metrics ---------------- #
        df_res = pd.read_csv(res_f)
        last   = df_res.iloc[-1]

        # Ultralytics results often store precision / recall but not F1.
        pr = last.get("metrics/precision(B)")
        rc = last.get("metrics/recall(B)")
        f1 = last.get("metrics/f1")  # some versions keep this
        if (pd.isna(f1) or f1 is None) and pr is not None and rc is not None:
            f1 = 2 * pr * rc / (pr + rc + 1e-9)

        # Wall-clock seconds: try args.yaml → best_hyperparameters.yaml
        seconds = yaml.safe_load(args_f.read_text()).get("time")
        if seconds is None:
            seconds = _ultra_seconds(ultra_root / "best_hyperparameters.yaml")

        metrics = {
            "f1_score":      f1,
            "precision":     pr,
            "recall":        rc,
            "mAP50":         last.get("metrics/mAP50(B)"),
            "mAP50_95":      last.get("metrics/mAP50-95(B)"),
            "last_epoch":    int(last.get("epoch", -1)),
            "execution_time": seconds,
            "memory_before": None,
            "memory_after":  None,
        }
        # ---------------- append row ---------------- #
        rows.append(
            TrialRow(
                model_family=family,
                size=size,
                optimiser="ultralytics_es",
                trial_id=trial_id,
                params=params,
                metrics=metrics,
            )
        )
    return rows

def _find_ultra_root(opt_dir: Path) -> Path | None:
    """Return the 'root' that contains many trainXX folders."""
    if (opt_dir / "train").is_dir():                  # one-level layout
        return opt_dir
    # fallback: first folder that has >10 'train\d+' dirs
    candidates = [p for p in opt_dir.iterdir() if p.is_dir()]
    for c in candidates:
        if sum(d.name.startswith("train") for d in c.iterdir()) > 10:
            return c
    return None


def _ultra_seconds(yaml_f: Path) -> float | None:
    if not yaml_f.exists():
        return None
    first = yaml_f.open("r", encoding="utf-8").readline()
    m = _ITER_LINE.search(first)
    return float(m.group(1)) if m else None


# ---------- Optuna ----------------------------------------------------------

def _collect_optuna_rows(opt_dir: Path, family: str, size: str, optimiser: str) -> List[TrialRow]:
    rows: List[TrialRow] = []
    try:
        db_file = next(opt_dir.glob("*.db"))
    except StopIteration:
        LOGGER.warning("No .db file in %s", opt_dir)
        return rows

    storage = f"sqlite:///{db_file}"
    study   = optuna.load_study(study_name=db_file.stem, storage=storage)

    mem_before, mem_after = _optuna_memory_stats(db_file)

    for t in study.get_trials(deepcopy=False):
        params = _filter_hp(t.params)
        metrics = {
            "f1_score":      t.value,
            "precision":     t.user_attrs.get("precision"),
            "recall":        t.user_attrs.get("recall"),
            "mAP50":         t.user_attrs.get("mAP50"),
            "mAP50_95":      t.user_attrs.get("mAP50_95"),
            "last_epoch":    t.user_attrs.get("last_epoch"),
            "execution_time": (t.datetime_complete - t.datetime_start).total_seconds()
                              if t.datetime_complete and t.datetime_start else None,
            "memory_before": mem_before.get(t.number),
            "memory_after":  mem_after.get(t.number),
        }
        rows.append(TrialRow(family, size, optimiser,
                             f"trial{t.number}", params, metrics))
    return rows


def _optuna_memory_stats(db_path: Path) -> tuple[Dict[int, int], Dict[int, int]]:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    mem_before: Dict[int, int] = {}
    mem_after:  Dict[int, int] = {}
    cur.execute("SELECT trial_id, key, value_json FROM trial_system_attributes")
    for tid, key, val in cur.fetchall():
        if key == "memory_before":
            mem_before[tid] = int(json.loads(val))
        elif key == "memory_after":
            mem_after[tid] = int(json.loads(val))
    con.close()
    return mem_before, mem_after


# ---------- shared ----------------------------------------------------------

def _filter_hp(raw: Dict[str, object]) -> Dict[str, float | int | str | None]:
    safe = {k: None for k in HP_COLS}
    for k in HP_COLS:
        if k in raw:
            safe[k] = raw[k]
    return safe
