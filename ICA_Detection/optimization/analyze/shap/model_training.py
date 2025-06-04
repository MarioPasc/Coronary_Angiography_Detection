"""
Train LightGBM surrogates & compute Tree-SHAP values.

Public entry-point
------------------
>>> train_and_save_models(df_trials, out_dir=Path("./outputs"))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import dump
import shap

from ICA_Detection.optimization.analyze.shap.constants import HP_COLS, MODEL_SIZES

LOGGER = logging.getLogger(__name__)

__all__ = ["train_and_save_models"]


def train_and_save_models(
    df: pd.DataFrame,
    out_dir: Path,
    cat_features: Sequence[str] = ("optimizer",),
) -> None:
    """
    Fit one LightGBM regressor per (model_family, size), compute SHAP, and save
    everything needed for visualisations.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    importance_records = []

    for family in ("yolo", "dca_yolo"):
        for size in MODEL_SIZES:
            subset = (df[(df["model_family"] == family) & (df["size"] == size)].reset_index(drop=True))          # guarantees contiguous ordering
            if subset.empty:
                LOGGER.warning("No trials for %s-%s – skipped", family, size)
                continue

            X = subset[HP_COLS].copy()
            y = subset["f1_score"].astype(float)

            # Cast categorical(s) – notably 'optimizer' from Ultralytics trials
            for col in cat_features:
                X[col] = X[col].astype("category")

            model = lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=-1,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                min_gain_to_split=0.0,        # quiet “-inf gain” warnings
            ).fit(X, y, categorical_feature=cat_features)

            # ── serialise model
            model_f = out_dir / f"{family}_{size}_model.txt"
            model.booster_.save_model(model_f)
            LOGGER.info("Model saved → %s", model_f.name)

            # ── SHAP values
            expl = shap.TreeExplainer(model.booster_)
            shap_vals = expl.shap_values(X, check_additivity=False)
            shap_f = out_dir / f"{family}_{size}_shap.npy"
            np.save(shap_f, shap_vals)
            LOGGER.info("SHAP saved → %s  (shape=%s)",
                        shap_f.name, shap_vals.shape)

            # ── design matrix
            X_f = out_dir / f"{family}_{size}_X.parquet"
            X.to_parquet(X_f)

            # ── global importance
            imp = np.abs(shap_vals).mean(axis=0)
            imp /= imp.sum()
            importance_records.extend(
                {
                    "model_family": family,
                    "size": size,
                    "hyperparameter": hp,
                    "importance": float(val),
                }
                for hp, val in zip(HP_COLS, imp)
            )

    pd.DataFrame(importance_records).to_parquet(
        out_dir / "global_importance.parquet", index=False
    )
    LOGGER.info("Global importance table written (%d rows)",
                len(importance_records))