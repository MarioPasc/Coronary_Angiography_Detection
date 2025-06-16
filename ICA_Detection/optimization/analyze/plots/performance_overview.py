#!/usr/bin/env python3
"""
performance_overview.py
=======================

Publication-grade comparison of optimisation strategies for YOLOv8 and
DCA-YOLOv8 using only the first N Optuna trials per run (default N = 100).
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import optuna
import pandas as pd
from scipy import stats
import scienceplots
import argparse

plt.style.use(["science", "ieee", "grid"])

# ─────────────────────  CONFIG & GLOBAL CONSTANTS  ──────────────────────── #

PARETO_COLOR = "#000000"
GRID_STYLE   = dict(linestyle=":", linewidth=0.5, alpha=0.75)

# ── NEW: how many Optuna trials to keep (1 … ∞) ─────────────────────────── #
FIRST_N_TRIALS = 100  # ← change here or pass --n-trials on CLI

_OPTIMISER_COLOURS: Dict[str, str] = {
    "cmaes":          "#0072B2",
    "gpsampler":      "#56B4E9",
    "random":         "#009E73",
    "tpe":            "#E69F00",
    "ultralytics_es": "#D55E00",
}
_MODEL_MARKERS = {"v8s": "D", "v8m": "s", "v8l": "^"}

# ─────────────────────────────  LOGGING  ─────────────────────────────────── #

LOGGER = logging.getLogger("perf-overview")
LOGGER.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
LOGGER.addHandler(_handler)

# ────────────────────────  DATA STRUCTURES  ─────────────────────────────── #

OptimiserName = Literal["cmaes", "gpsampler", "random", "tpe", "ultralytics_es"]
ModelSize     = Literal["v8s", "v8m", "v8l"]

@dataclass(frozen=False, slots=True)
class OptimisationResult:
    model: str
    optimiser: OptimiserName
    total_seconds: float          # raw wall-clock time
    best_f1: float
    n_gpus: int = 3               # NEW  (defaults to 3)
    speed_up: float = 1.0         # filled later

# ─────────────────────────  PUBLIC ENTRY  ───────────────────────────────── #

def generate_plots(
    root: Path | str,
    plot_root:  Path | str,
    file_format: str = "pdf",
    cv_csv: Path | str | None = None,
    gpu_csv: Path | str | None = None,       # NEW
    n_trials: int = FIRST_N_TRIALS,                 # ← NEW PARAM
) -> None:
    """
    Parameters
    ----------
    root : root folder with yolov8*/ and dca_yolov8*/ sub-dirs
    plot_root   : destination; <plot_root>/performance/ created if needed
    file_format : «pdf», «png», …
    cv_csv      : optional CSV with columns  model_variant, optimizer, f1_score
    n_trials    : analyse only the first *n* Optuna trials  (default 100)
    """
    root = Path(root).expanduser().resolve()
    out_dir     = Path(plot_root).expanduser().resolve() / "performance"
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_map  = _read_gpu_map(gpu_csv)
    results = _collect_all_results(root, n_trials)
    cv_stats = _read_cv_stats(cv_csv) if cv_csv else None

    # attach GPU counts + normalise run times to 3-GPU baseline
    for r in results:
        r.n_gpus = gpu_map.get((r.model, r.optimiser), 3)
        r.total_seconds *= r.n_gpus / 3          # <-- key line

    # split groups ────────────────────────────────────────────────── #
    yolo_res = [r for r in results if not r.model.startswith("dca_")]
    dca_res  = [r for r in results if     r.model.startswith("dca_")]

    # composite figure ────────────────────────────────────────────── #
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(18, 10))
    panel_cfgs: List[Tuple[str, Iterable[OptimisationResult]]] = [
        ("a.", yolo_res), ("b.", dca_res)
    ]

    ymin, ymax = _global_y_limits(results, cv_stats)
    for ax, (label, subset) in zip(axes, panel_cfgs):
        _populate_panel(ax, subset, label, cv_stats, ymin, ymax)

    _build_joint_legends(fig)
    fig.subplots_adjust(top=0.82, bottom=0.25, wspace=0.12)

    out_file = out_dir / f"performance_overview.{file_format}"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    LOGGER.info("Saved %s", out_file.relative_to(out_dir.parent))
    plt.close(fig)

# ─────────────────────────────  HELPERS  ─────────────────────────────────── #

def _read_gpu_map(csv_path: Path | str | None
                  ) -> dict[tuple[str, str], int]:
    if csv_path is None:
        return {}
    df = pd.read_csv(csv_path)
    return {(m, o): int(g) for m, o, g in df.itertuples(index=False)}


def _read_cv_stats(csv_path: Path | str | None
                   ) -> dict[Tuple[str, str], Tuple[float, float]]:
    df = pd.read_csv(csv_path)
    groups = df.groupby(["model_variant", "optimizer"])["f1_score"]
    out: dict[Tuple[str, str], Tuple[float, float]] = {}
    for key, series in groups:
        mean = series.mean()
        sem  = stats.sem(series, ddof=1)
        ci95 = stats.t.ppf(0.975, len(series)-1) * sem
        out[key] = (mean, ci95)
    LOGGER.info("Loaded CV statistics for %d optimiser×model combos", len(out))
    return out

# ────────────────  OPTUNA HELPERS with TRIAL LIMIT  ─────────────────────── #

def _optuna_stats_first_n(db: Path, n: int) -> Tuple[float, float]:
    """
    Return (wall-clock seconds, best_f1) considering **only the first `n`
    completed trials** in an Optuna SQLite database.

    Logic
    -----
    1. Identify the `trial_id`s of the first *n* trials (chronological order).
    2. In table `trial_values`, pick the *single-objective* value with the
       highest `value` among those trials → that is the best F1.
    3. Compute the wall-clock time span between the earliest `datetime_start`
       and the latest `datetime_complete` within the same ID subset.
    """
    with sqlite3.connect(str(db)) as con:
        # 1 ── first n trial IDs
        trial_ids = [row[0] for row in con.execute(
            "SELECT trial_id FROM trials ORDER BY trial_id ASC LIMIT ?;",
            (n,)
        )]
        if not trial_ids:
            raise RuntimeError("No trials found in the database")

        ph     = ",".join("?" * len(trial_ids))          # SQL placeholders
        params = trial_ids                                # bind list → tuple automatically

        # 2 ── best F1 score among those IDs
        best_row = con.execute(
            f"""
            SELECT trial_id, value
            FROM trial_values
            WHERE trial_id IN ({ph})
              AND value IS NOT NULL
            ORDER BY value DESC
            LIMIT 1;
            """, params
        ).fetchone()
        if best_row is None:
            raise RuntimeError("No objective values found in the first N trials")

        best_value: float = float(best_row[1])

        # 3 ── wall-clock span of the same ID subset
        t0_str, tN_str = con.execute(
            f"""
            SELECT MIN(datetime_start), MAX(datetime_complete)
            FROM trials
            WHERE trial_id IN ({ph});
            """, params
        ).fetchone()

    t0 = datetime.fromisoformat(t0_str)
    tN = datetime.fromisoformat(tN_str)
    total_seconds = (tN - t0).total_seconds()

    return total_seconds, best_value



# ────────────────  OTHER PARSERS (unchanged) ────────────────────────────── #

_ITER_RE = re.compile(r"\(\s*([0-9.]+)s\)")

def _ultra_total_seconds(yaml_f: Path) -> float:
    with yaml_f.open(encoding="utf-8") as fh:
        line = fh.readline()
    m = _ITER_RE.search(line)
    if not m:
        raise RuntimeError(f"No runtime in {yaml_f}")
    return float(m.group(1))

def _ultra_best_f1(folder: Path) -> float:
    from ICA_Detection.optimization.analyze.fetch_best_trial import _scan_ultralytics_es
    _, best, *_ = _scan_ultralytics_es(folder.parent)
    return float(best)
# ────────────────  DIRECTORY TRAVERSAL  ─────────────────────────────────── #

def _collect_all_results(root: Path, n_trials: int) -> List[OptimisationResult]:
    out: List[OptimisationResult] = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for opt_dir in model_dir.iterdir():
            if opt_dir.name not in _OPTIMISER_COLOURS:
                continue
            opt: OptimiserName = opt_dir.name  # type: ignore[assignment]
            try:
                if opt == "ultralytics_es":
                    sub = opt_dir / "ultralytics_es"
                    secs = _ultra_total_seconds(sub / "best_hyperparameters.yaml")
                    best = _ultra_best_f1(sub)
                else:
                    db_file = next(opt_dir.glob("*.db"))
                    secs, best = _optuna_stats_first_n(db_file, n_trials)
            except Exception as exc:          # pragma: no cover
                LOGGER.warning("Skipping %s / %s: %s", model_name, opt, exc)
                continue
            out.append(OptimisationResult(model_name, opt, secs, best))
    LOGGER.info("Collected %d optimiser×model results", len(out))
    return out

# ─────────────────────────────  PLOTTING  ────────────────────────────────── #

def _global_y_limits(results: Iterable[OptimisationResult],
                     cv_stats: dict | None) -> Tuple[float, float]:
    vals = []
    for r in results:
        ci = cv_stats.get((r.model, r.optimiser), (0, 0))[1] if cv_stats else 0
        vals.extend([r.best_f1 - ci, r.best_f1 + ci])
    ymin, ymax = min(vals), max(vals)
    pad = 0.01
    return ymin - pad, ymax + pad

def _populate_panel(ax: plt.Axes,
                    results: Iterable[OptimisationResult],
                    panel_label: str,
                    cv_stats: dict | None,
                    ymin: float, ymax: float) -> None:
    if not results:
        ax.axis("off")
        return

    # ── speed-up values and x-axis limits ─────────────────────────── #
    slowest = max(r.total_seconds for r in results)
    for r in results:
        r.speed_up = slowest / r.total_seconds

    right_bound = 2 ** np.ceil(np.log2(max(r.speed_up for r in results)))  # next power-of-2

    # ── scatter + error bars ──────────────────────────────────────── #
    for r in results:
        size_tag: ModelSize = r.model.rsplit("yolo", 1)[-1]
        x, y = r.speed_up, r.best_f1

        ax.scatter(x, y,
                   marker=_MODEL_MARKERS[size_tag],
                   color=_OPTIMISER_COLOURS[r.optimiser],
                   edgecolors="black", s=70, linewidths=0.6, zorder=3)

        if cv_stats and (r.model, r.optimiser) in cv_stats:
            _, ci95 = cv_stats[(r.model, r.optimiser)]
            ax.errorbar(x, y, yerr=ci95, fmt="none",
                        ecolor="black", capsize=3, elinewidth=0.8, zorder=2.9)

    # ── step connectors per optimiser ─────────────────────────────── #
    for opt in _OPTIMISER_COLOURS:
        pts = sorted((r for r in results if r.optimiser == opt),
                     key=lambda r: r.speed_up)
        if len(pts) < 2:
            continue
        xs, ys = [], []
        xp, yp = pts[0].speed_up, pts[0].best_f1
        for nxt in pts[1:]:
            xn, yn = nxt.speed_up, nxt.best_f1
            xs.extend([xp, xp, xn])
            ys.extend([yp, yn, yn])
            xp, yp = xn, yn
        ax.plot(xs, ys, color=_OPTIMISER_COLOURS[opt],
                linewidth=1.0, alpha=0.45, zorder=2)

    # ── Pareto frontier (non-dominated: larger speed-up *and* higher F1) ─ #
    # Traverse **from right to left** (highest → lowest speed-up).  Keep a
    # point if its F1 is strictly better than any previously seen; this
    # guarantees non-domination in both coordinates.
    pts_sorted = sorted(results, key=lambda r: r.speed_up, reverse=True)
    frontier_pairs = []
    best_f1_so_far = -np.inf
    for r in pts_sorted:
        if r.best_f1 > best_f1_so_far + 1e-12:
            frontier_pairs.append((r.speed_up, r.best_f1))
            best_f1_so_far = r.best_f1

    if frontier_pairs:
        # Reverse to ascending x-order for plotting
        frontier_x, frontier_y = zip(*reversed(frontier_pairs))

        frontier_x_ext = (0.9,) + frontier_x + (right_bound,)
        frontier_y_ext = (frontier_y[0],) + frontier_y + (frontier_y[-1],)

        ax.step(frontier_x_ext, frontier_y_ext, where="post",
                color=PARETO_COLOR, linewidth=2,
                label="Pareto frontier", zorder=4, linestyle="-")

        ax.fill_between(frontier_x_ext, frontier_y_ext, y2=ymin,
                        step="post", color=PARETO_COLOR,
                        alpha=0.05, zorder=1)

    # ── axis cosmetics ─────────────────────────────────────────────── #
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.9, right_bound)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=2))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):d}×"))

    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("F1-score", fontsize=22) if panel_label == "a." else None
    ax.set_xlabel(r"Speed-up factor ($log_2$)", fontsize=22)
    ax.tick_params(axis='both', labelsize=20)  # Increased x and y-tick label size
    ax.text(-0.03, 1.03, panel_label, transform=ax.transAxes,
            fontsize=22, fontweight="bold")

    ax.grid(True, **GRID_STYLE)
    ax.spines[['right', 'top']].set_visible(False)


def _build_joint_legends(fig: plt.Figure) -> None:
    # colour legend  + Pareto entry  (bottom)
    colour_handles = [
        plt.Line2D([], [], marker="o", linestyle="none",
                   markerfacecolor=col, markeredgecolor="black",
                   markersize=8, label=opt.upper().replace("_ES", "-ES"))
        for opt, col in _OPTIMISER_COLOURS.items()
    ]
    pareto_handle = plt.Line2D([], [], color=PARETO_COLOR, linewidth=2,
                               label="Pareto frontier")
    fig.legend(handles=colour_handles + [pareto_handle],
               loc="lower center", bbox_to_anchor=(0.5, 0.04),
               ncol=(len(colour_handles) + 1)/2, frameon=False, fontsize=22)

    # shape legend (models)  (top)
    marker_handles = [
        plt.Line2D([], [], linestyle="none",
                   marker=_MODEL_MARKERS[size],
                   markerfacecolor="white", markeredgecolor="black",
                   markersize=8, label=f"{size}")
        for size in ["v8s", "v8m", "v8l"]
    ]
    fig.legend(handles=marker_handles,
               loc="upper center", bbox_to_anchor=(0.5, 0.95),
               ncol=len(marker_handles), frameon=False, fontsize=22)

DEFAULT_FIRST_N_TRIALS: int = 100
# ─────────────────────────────  CLI  ─────────────────────────────────────── #

if __name__ == "__main__":
    DEFAULT_BASE_DIR_CLI = Path("/media/mpascual/PortableSSD/Coronariografías/CompBioMed/bho_compbiomed/cadica")
    DEFAULT_FMT_CLI = "pdf"
    # DEFAULT_FIRST_N_TRIALS is already defined globally

    parser = argparse.ArgumentParser(description="Generate performance overview plots. Compares optimisation strategies for YOLOv8 and DCA-YOLOv8.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR_CLI,
                        help=f"Root directory containing 'optimization', 'kfold', and 'gpu_usage_combined' subdirectories. (default: {DEFAULT_BASE_DIR_CLI})")
    parser.add_argument("--fmt", type=str, default=DEFAULT_FMT_CLI,
                        help=f"Output image format (e.g., pdf, png, svg). (default: {DEFAULT_FMT_CLI})")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_FIRST_N_TRIALS,
                        help=f"Analyze only the first N completed Optuna trials per study. (default: {DEFAULT_FIRST_N_TRIALS})")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose console output (DEBUG level).")
    
    args = parser.parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
        for handler in LOGGER.handlers: # Ensure all handlers respect the new level
            handler.setLevel(logging.DEBUG)
        LOGGER.debug("Verbose logging enabled.")

    LOGGER.info("Running performance_overview.py as a script.")
    LOGGER.info("Expected folder structure under --base-dir:")
    LOGGER.info("  <base_dir>/optimization/<model_name>/<optimizer_name>/")
    LOGGER.info("  <base_dir>/kfold/kfold_metrics.csv (optional)")
    LOGGER.info("  <base_dir>/gpu_usage_combined/gpu_usage_combined.csv (optional)")
    
    optimization_path = args.base_dir / "optimization"
    cv_csv_path = args.base_dir / "kfold" / "kfold_metrics.csv"
    gpus_csv_path = args.base_dir / "gpu_usage_combined" / "gpu_usage_combined.csv"
    figures_out_dir = args.base_dir / "figures" # Figures will be in <base_dir>/figures/performance/

    # Check existence of optional files and inform user
    if not cv_csv_path.exists():
        LOGGER.info(f"Optional CV metrics file not found: {cv_csv_path}. Proceeding without CV error bars.")
        cv_csv_path = None # Set to None if not found
    if not gpus_csv_path.exists():
        LOGGER.info(f"Optional GPU usage file not found: {gpus_csv_path}. Proceeding with default GPU assumptions for normalization.")
        gpus_csv_path = None # Set to None if not found


    generate_plots(
        root=optimization_path,
        plot_root=figures_out_dir,  # generate_plots will create a 'performance' subdir here
        file_format=args.fmt,
        cv_csv=cv_csv_path,
        gpu_csv=gpus_csv_path,
        n_trials=args.n_trials,
    )