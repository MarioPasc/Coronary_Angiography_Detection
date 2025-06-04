"""
Constant definitions shared by *data_ingestion* and *model_training* modules.
"""

from typing import Literal, Sequence

# ────────────────────────────────────────────────────────────────────────────
# Columns & categorizations
# ────────────────────────────────────────────────────────────────────────────

FAMILY_COLS = ["model_family", "size", "optimiser", "trial_id"]

HP_COLS = [
    "optimizer",           # choice inside Ultralytics training loop
    "batch",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_momentum",
    "box",
    "cls",
    "dfl",
]

METRIC_COLS = [
    "f1_score",
    "precision",
    "recall",
    "mAP50",
    "mAP50_95",
    "execution_time",
    "memory_before",
    "memory_after",
    "last_epoch",
]

ALL_COLS = FAMILY_COLS + HP_COLS + METRIC_COLS

# ────────────────────────────────────────────────────────────────────────────
# Other categorical enumerations
# ────────────────────────────────────────────────────────────────────────────

OPT_LIST: Sequence[str] = ("cmaes", "gpsampler", "random", "tpe", "ultralytics_es")
MODEL_SIZES: Sequence[str] = ("s", "m", "l")

DatasetFmt = Literal["csv", "parquet"]
