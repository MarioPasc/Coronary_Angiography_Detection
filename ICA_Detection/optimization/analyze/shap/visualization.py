"""
strategy_heatmaps.py
====================

Create *one* heat-map per optimiser **and** architecture:

    rows   → model sizes  (v8s, v8m, v8l)
    columns→ 11 hyper-parameters
    cells  → SHAP importance (0-100 %)

• 5 optimisers  ×  2 architectures  =  **10 figures**
• Each figure saved *without* colour-bar.
• A single colour-bar PDF (`heatmap_colorbar.<fmt>`) is written separately.

The function is meant to be imported **or** called via the standalone CLI
(`heatmap_cli.py`).

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ICA_Detection.optimization.analyze.shap.constants import HP_COLS, MODEL_SIZES, OPT_LIST, DICT_HYPERPARAMETERS

import scienceplots
plt.style.use(["science", "ieee"])  # use SciencePlots for better aesthetics

LOGGER = logging.getLogger(__name__)
sns.set_style("white")


# ────────────────────────────────────────────────────────────────────────────
# Public façade
# ────────────────────────────────────────────────────────────────────────────

def make_strategy_heatmaps(artifacts_dir: Path, out_dir: Path, fmt: str = "pdf") -> None:
    """
    Build ten optimiser-specific heat-maps + a separate colour-bar.

    Parameters
    ----------
    artifacts_dir
        Folder that already contains `shap_dataset.parquet`,
        `<family>_<size>_shap.npy`, and `<family>_<size>_X.parquet`.
    out_dir
        Where `heatmap_<family>_<optimiser>.pdf` and `heatmap_colorbar.<fmt>`
        will be written.
    fmt
        Graphic file format accepted by Matplotlib ('pdf', 'png', ...).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df_trials = pd.read_parquet(artifacts_dir / "shap_dataset.parquet")
    LOGGER.info("Loaded shap_dataset (%d rows)", len(df_trials))

    # ---------- collect importance per (family,size,optimiser) --------------
    recs: List[Dict[str, float | str]] = []

    for family in ("yolo", "dca_yolo"):
        for size in MODEL_SIZES:
            shp_f = artifacts_dir / f"{family}_{size}_shap.npy"
            if not shp_f.exists():
                LOGGER.debug("Missing %s – skipped", shp_f.name)
                continue
            shap_vals = np.load(shp_f)
            # optimiser labels for these rows
            mask = (df_trials["model_family"] == family) & (df_trials["size"] == size)
            opt_labels = df_trials.loc[mask, "optimiser"].values
            if len(opt_labels) > shap_vals.shape[0]:
                assert len(opt_labels) == shap_vals.shape[0]

            for opt in np.unique(opt_labels):
                rows = opt_labels == opt
                if not rows.any():
                    continue
                imp = np.abs(shap_vals[rows]).mean(axis=0)
                imp /= imp.sum() or 1.0  # avoid /0

                for hp, val in zip(HP_COLS, imp):
                    recs.append({
                        "model_family": family,
                        "size": size,
                        "optimiser": opt,
                        "hyperparameter": hp,
                        "importance": float(val),
                    })

    df_imp = pd.DataFrame(recs)
    vmax = df_imp["importance"].max() * 100  # common scale

    # ---------- draw heat-maps ---------------------------------------------
    for fam in ("yolo", "dca_yolo"):
        for opt in OPT_LIST:
            _heat_single(
                df_imp[(df_imp["model_family"] == fam) & (df_imp["optimiser"] == opt)],
                fam, opt, vmax, out_dir / f"heatmap_{fam}_{opt}.{fmt}"
            )

    # ---------- single colour-bar ------------------------------------------
    _save_colourbar(vmax, out_dir / f"heatmap_colorbar.{fmt}")
    LOGGER.info("All heat-maps & colour-bar written to %s", out_dir)


# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────

def _heat_single(df: pd.DataFrame, family: str, opt: str, vmax: float, path: Path) -> None:
    """
    Generate one heat-map: models (rows) × hyper-parameters (cols).
    """
    if df.empty:
        LOGGER.warning("No data for %s-%s – skipped heat-map", family, opt)
        return

    pivot = (
        df.pivot(index="size", columns="hyperparameter", values="importance")
          .reindex(index=MODEL_SIZES, columns=HP_COLS)
          .fillna(0.0) * 100
    )

    fig, ax = plt.subplots(figsize=(8, 1.6))
    sns.heatmap(pivot,
                ax=ax,
                cmap="viridis",
                vmin=0,
                vmax=vmax,
                cbar=False,
                linewidths=0.3,
                linecolor="white")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels([f"v8{s}" for s in MODEL_SIZES], rotation=0)
    # Use DICT_HYPERPARAMETERS for x-axis tick labels
    xticklabels = [DICT_HYPERPARAMETERS.get(hp, hp) for hp in pivot.columns]
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.set_title(f"{family.upper()} – {opt.upper()}", fontsize=9, pad=8)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Saved %s", path.name)


def _save_colourbar(vmax: float, path: Path) -> None:
    """
    Save a stand-alone vertical colour-bar matching the Viridis scale.
    """
    fig, ax = plt.subplots(figsize=(1.1, 3.6))
    norm = plt.Normalize(vmin=0, vmax=vmax)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
        cax=ax, orientation="vertical", label="Importance (%)"
    )
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Saved %s", path.name)
