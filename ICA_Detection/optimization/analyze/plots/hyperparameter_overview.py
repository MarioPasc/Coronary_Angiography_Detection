#!/usr/bin/env python3
"""
hyperparameter_overview.py
==========================

Creates two compact 1 × 2-subplot figures (YOLOv8 & DCA-YOLOv8) that
summarise the *hyper-parameter decisions* of five optimisation strategies.

Left panel : violin + box plot for the three key hyper-parameters  
Right panel: 2-D UMAP of the elite 10 % trials (+ 95 % ellipses)

Improvements vs. v1
-------------------
1.  **Ultralytics-ES reader fixed**
    • detects every `train*/args.yaml` in the `ultralytics_es/` folder  
    • pulls the matching `results.csv` to extract the best F1

2.  **Cleaner violin labels**
    • param name + p-value star centred above its own row (no big left offset)

3.  **Figure cosmetics**
    • no global title; sub-titles reduced to “a.” and “b.”  
    • *shape* legend (network size) sits top-centre  
    • *colour* legend (optimiser) sits immediately beneath

Everything that already worked is left untouched.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import optuna
import pandas as pd
from umap import UMAP
from matplotlib import ticker
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

# ─────────────────────────── CONFIG ───────────────────────────────────── #

import scienceplots

plt.style.use(["science", "ieee", "grid"])

LOGGER = logging.getLogger("hyperparam-decisions")
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler())

FIRST_N_TRIALS = 100
ELITE_FRAC = 0.10                      # fraction of best trials kept per group
TOP_PARAMS = ["lr0", "lrf", "momentum", "weight_decay", "box", "dfl"]

_OPTIMISER_COLOURS: Dict[str, str] = {
    "cmaes":          "#0072B2",
    "gpsampler":      "#56B4E9",
    "random":         "#009E73",
    "tpe":            "#E69F00",
    "ultralytics_es": "#D55E00",
}
_MODEL_MARKERS = {"v8s": "D", "v8m": "s", "v8l": "^"}

OptimiserName = Literal[
    "cmaes", "gpsampler", "random", "tpe", "ultralytics_es"
]

# ────────────────────────── DATA CLASS ─────────────────────────────────── #


@dataclass(slots=True)
class TrialRecord:
    model: str                 # e.g. yolov8s  OR  dca_yolov8m
    optimiser: OptimiserName
    size: str                  # v8s / v8m / v8l
    f1: float
    params: Dict[str, float]


# ────────────────────── DATA ACCESS FUNCTIONS ─────────────────────────── #


def _read_optuna_trials(db: Path, first_n: int) -> List[TrialRecord]:
    """Load up to *first_n* completed trials from an Optuna SQLite DB."""
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{db}")
    study = optuna.load_study(study_name=str(db.stem), storage=storage)

    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE][:first_n]

    out: List[TrialRecord] = []
    for t in trials:
        out.append(
            TrialRecord(
                model=_infer_model_variant(db),
                optimiser=_infer_optimiser(db),
                size=_infer_size(db),
                f1=float(t.values[0]),
                params={k: float(v) for k, v in t.params.items()
                        if k in TOP_PARAMS},
            )
        )
    return out


def _read_ultra_trials(root: Path, first_n: int) -> List[TrialRecord]:
    """
    Ultralytics-ES layout:

    model/ultralytics_es/
        ├── train/
        ├── train1/
        ├── train2/
        └── …/args.yaml + results.csv
    """
    train_dirs = sorted(p for p in root.glob("train*") if p.is_dir())[:first_n]
    records: List[TrialRecord] = []

    for td in train_dirs:
        args_file = td / "args.yaml"
        res_file = td / "results.csv"
        if not args_file.exists() or not res_file.exists():
            continue

        with args_file.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        # --- best F1 from results.csv ---------------------------------------- #
        try:
            df_res = pd.read_csv(res_file)
            f1_cols = [c for c in df_res.columns if c.lower().endswith("f1")]
            fitness_cols = [c for c in df_res.columns
                            if "fitness" in c.lower()]
            if f1_cols:
                f1_val = df_res[f1_cols[-1]].max()
            elif fitness_cols:
                f1_val = df_res[fitness_cols[-1]].max()
            else:
                f1_val = np.nan
        except Exception as exc:
            LOGGER.warning("Could not read %s: %s", res_file, exc)
            f1_val = np.nan

        params = {k: float(cfg[k]) for k in TOP_PARAMS if k in cfg}

        records.append(
            TrialRecord(
                model=_infer_model_variant(root),
                optimiser="ultralytics_es",
                size=_infer_size(root),
                f1=float(f1_val),
                params=params,
            )
        )
    return records


# ──────────── HELPERS TO GUESS MODEL / SIZE / OPTIMISER ──────────────── #


def _infer_optimiser(p: Path) -> OptimiserName:
    for opt in _OPTIMISER_COLOURS:
        if opt in p.parts:
            return opt  # type: ignore[return-value]
    raise RuntimeError(f"Cannot infer optimiser from {p}")


def _infer_model_variant(p: Path) -> str:
    for part in p.parts:
        if part.startswith("dca_") or part.startswith("yolo"):
            return part
    return "UNKNOWN"


def _infer_size(p: Path) -> str:
    return _infer_model_variant(p).split("yolo")[-1]  # v8s / v8m / v8l


# ───────────────────────── CORE LOADING ───────────────────────────────── #


def collect_trials(base_dir: Path,
                   first_n: int = FIRST_N_TRIALS) -> List[TrialRecord]:
    """Traverse *base_dir* and collate trials for all strategies."""
    recs: List[TrialRecord] = []

    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        for opt_dir in model_dir.iterdir():
            if opt_dir.name not in _OPTIMISER_COLOURS:
                continue
            try:
                if opt_dir.name == "ultralytics_es":
                    recs.extend(_read_ultra_trials(opt_dir, first_n))
                else:
                    db_file = next(opt_dir.glob("*.db"))
                    recs.extend(_read_optuna_trials(db_file, first_n))
            except Exception as exc:
                LOGGER.warning("Skipping %s: %s", opt_dir, exc)

    LOGGER.info("Loaded %d trial records", len(recs))
    return recs


# ────────────────── PANEL A  (VIOLIN / BOX) ───────────────────────────── #


def violin_panel(ax: plt.Axes,
                 df: pd.DataFrame,
                 param_names: List[str]) -> None:
    """Row-stacked violin/box plot (+ p-value stars)."""
    strategies = df["optimiser"].unique().tolist()
    width = 0.8
    row_gap = len(strategies) + 1

    for r, par in enumerate(param_names):
        base = r * row_gap

        # one violin per optimiser
        for j, opt in enumerate(strategies):
            vals = df.loc[df["optimiser"] == opt, par].dropna()
            if vals.empty:
                continue
            parts = ax.violinplot(vals.values,
                                  positions=[base + j],
                                  widths=width,
                                  showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(_OPTIMISER_COLOURS[opt])
                pc.set_edgecolor("black")
                pc.set_alpha(0.6)

            med = vals.median()
            q1, q3 = np.percentile(vals, [25, 75])
            ax.plot([base + j - width/2, base + j + width/2], [med, med],
                    color="black", lw=1.1)
            ax.add_patch(
                mpatches.Rectangle((base + j - width/2, q1),
                                   width, q3 - q1,
                                   fill=False, edgecolor="black", lw=0.8)
            )

        # significance star
        groups = [df.loc[df["optimiser"] == o, par].dropna().values
                  for o in strategies]
        if all(len(g) > 1 for g in groups):
            _, p_val = stats.kruskal(*groups)
        else:
            p_val = 1.0
        star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else \
               "*" if p_val < 0.05 else ""
        x_center = base + (len(strategies) - 1) / 2
        y_max = df[par].max()
        ax.text(x_center, y_max * 1.1,
                f"{par}{star}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xlim(-1, row_gap * len(param_names) - 0.5)
    ax.set_xticks([])
    ax.set_ylabel("Value (log-scale)")
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)


# ────────────────── PANEL B  (UMAP + ELLIPSES) ────────────────────────── #


def umap_panel(ax: plt.Axes,
               df: pd.DataFrame,
               param_names: List[str]) -> None:
    """UMAP embedding + 95 % confidence ellipses."""
    df = df.copy()
    df["opt_cat"] = df["optimiser"]

    ct = ColumnTransformer([
        ("num", StandardScaler(), param_names),
        ("cat", OneHotEncoder(), ["opt_cat"]),
    ])
    X = ct.fit_transform(df)

    embed = UMAP(n_neighbors=15, min_dist=0.3,
                 metric="euclidean", random_state=0).fit_transform(X)
    df["_x"], df["_y"] = embed[:, 0], embed[:, 1]

    for opt in df["optimiser"].unique():
        sub = df[df["optimiser"] == opt]
        # scatter per network size
        for sz in sub["size"].unique():
            pts = sub[sub["size"] == sz]
            ax.scatter(pts["_x"], pts["_y"],
                       marker=_MODEL_MARKERS[sz],
                       facecolor=_OPTIMISER_COLOURS[opt],
                       edgecolor="black", linewidth=0.4,
                       s=50, alpha=0.8, zorder=3)

        # 95 % ellipse
        if len(sub) >= 3:
            xy = sub[["_x", "_y"]].values
            cov = np.cov(xy, rowvar=False)
            mean = xy.mean(axis=0)
            evals, evecs = np.linalg.eigh(cov)
            order = evals.argsort()[::-1]
            evals, evecs = evals[order], evecs[:, order]
            chisq = stats.chi2.ppf(0.95, 2)
            w, h = 2 * np.sqrt(evals * chisq)
            ang = math.degrees(math.atan2(*evecs[:, 0][::-1]))
            ell = mpatches.Ellipse(mean, w, h, angle=ang,
                                   facecolor=_OPTIMISER_COLOURS[opt],
                                   edgecolor="black", lw=0.7,
                                   alpha=0.15, zorder=2)
            ax.add_patch(ell)

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)


# ───────────────────────  FIGURE DRIVER  ──────────────────────────────── #


def build_one_figure(df_model: pd.DataFrame,
                     out_path: Path,
                     fmt: str) -> None:
    """One 1 × 2 figure (violin + UMAP) for a given model family."""
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(11, 4.0), dpi=300,
        sharey=False, constrained_layout=True
    )

    # panel A
    violin_panel(axA, df_model, TOP_PARAMS)
    axA.set_title("a.", loc="left", fontsize=12, pad=2)

    # panel B
    umap_panel(axB, df_model, TOP_PARAMS)
    axB.set_title("b.", loc="left", fontsize=12, pad=2)

    # legends -------------------------------------------------------------- #
    optimiser_names = list(_OPTIMISER_COLOURS.keys())
    colour_handles = [
        Line2D([], [], marker="o", linestyle="",
               markerfacecolor=_OPTIMISER_COLOURS[o],
               markeredgecolor="black", markersize=8,
               label=o.upper().replace("_ES", "-ES")) # Label is already set here
        for o in optimiser_names
    ]
    colour_labels = [h.get_label() for h in colour_handles]

    model_sizes = ["v8s", "v8m", "v8l"]
    shape_handles = [
        Line2D([], [], marker=_MODEL_MARKERS[s], linestyle="",
               markerfacecolor="white", markeredgecolor="black",
               markersize=8, label=s) # Label is already set here
        for s in model_sizes
    ]
    shape_labels = [h.get_label() for h in shape_handles]

    if "dca" in out_path.stem: 
        # colours legend – just below shapes
        fig.legend(colour_handles, colour_labels, ncol=len(colour_handles),
                loc="upper center", bbox_to_anchor=(0.5, 0.03),
                frameon=False, fontsize=9)
    else:
            # shapes legend – top centre
        fig.legend(shape_handles, shape_labels, ncol=3,
                loc="upper center", bbox_to_anchor=(0.5, 1.08),
                frameon=False, fontsize=9)
    
    axB.spines[['right', 'top']].set_visible(False)

    # save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Figure saved → %s", out_path.with_suffix(f'.{fmt}'))


# ─────────────────────────────  MAIN  ─────────────────────────────────── #


def main() -> None:
    ap = argparse.ArgumentParser(description="Hyper-parameter decision plots")
    ap.add_argument("--base-dir", required=True, type=Path,
                    help="Root dir with yolov8*/ and dca_yolov8*/ sub-dirs")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Destination directory for the figures")
    ap.add_argument("--fmt", default="pdf",
                    help="Output format: pdf / png / …")
    args = ap.parse_args()

    trials = collect_trials(args.base_dir, first_n=FIRST_N_TRIALS)
    if not trials:
        LOGGER.error("No trial data found – check --base-dir")
        return

    # dataframe
    df = pd.DataFrame([{
        "model": t.model,
        "optimiser": t.optimiser,
        "size": t.size,
        "f1": t.f1,
        **{k: v for k, v in t.params.items()}
    } for t in trials])

    # choose elite
    elite_chunks = []
    for _, grp in df.groupby(["model", "size", "optimiser"]):
        k = max(1, int(len(grp) * ELITE_FRAC))
        elite_chunks.append(grp.nlargest(k, "f1"))
    elite = pd.concat(elite_chunks, ignore_index=True)

    # split by model family
    base_mask = ~elite["model"].str.startswith("dca_")
    df_yolo = elite[base_mask]
    df_dca = elite[~base_mask]

    out_root = args.out_dir
    build_one_figure(df_yolo, out_root / "hyperparams_yolov8", args.fmt)
    build_one_figure(df_dca, out_root / "hyperparams_dca_yolov8", args.fmt)


if __name__ == "__main__":
    main()
