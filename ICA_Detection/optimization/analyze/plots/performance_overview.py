"""
performance_overview.py
=======================

Generate runtime-vs-F1 scatter plots (plus stand-alone legends) for the
CADiCA hyper-parameter sweeps on YOLOv8 and DCA-YOLOv8.

Four files are written to  <plots_root>/performance/ :

  • yolo_performance.<fmt>            – base YOLO models only
  • dca_yolo_performance.<fmt>        – DCA-YOLO models only
  • colour_legend.<fmt>               – optimiser → colour mapping
  • marker_legend.<fmt>               – model-size → marker mapping
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Literal

import matplotlib.pyplot as plt
import optuna
import numpy as np

import scienceplots
plt.style.use(["science", "ieee", "grid"])

# --------------------------------------------------------------------------- #
# Logger configuration
# --------------------------------------------------------------------------- #

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
LOGGER.addHandler(_handler)

# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

OptimiserName = Literal["cmaes", "gpsampler", "random", "tpe", "ultralytics_es"]
ModelSize = Literal["v8s", "v8m", "v8l"]


@dataclass(frozen=True, slots=True)
class OptimisationResult:
    """Container for one optimiser × one model variant."""

    model: str                   # e.g. "yolov8s" or "dca_yolov8m"
    optimiser: OptimiserName
    total_seconds: float
    best_f1: float


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def generate_plots(
    cadica_root: Path | str,
    plot_root: Path | str,
    file_format: str = "pdf",
) -> None:
    """
    Crawl *cadica_root* and write four figures under *plot_root*/performance/.

    Parameters
    ----------
    cadica_root
        Folder containing `yolov8{s,m,l}` and `dca_yolov8{s,m,l}` sub-folders.
    plot_root
        Where `<plot_root>/performance/` is created (if absent).
    file_format
        Extension accepted by ``matplotlib`` (default: ``pdf``).
    """
    cadica_root = Path(cadica_root).expanduser().resolve()
    plot_root = Path(plot_root).expanduser().resolve()
    perf_dir = plot_root / "performance"
    perf_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing plots to %s", perf_dir)

    results = _collect_all_results(cadica_root)

    # Determine overall maximum time for consistent x-axis scaling
    if not results:
        overall_max_hours = 20.0  # Default if no results (e.g., show a 0-20h range)
    else:
        max_total_seconds = max(r.total_seconds for r in results)
        if max_total_seconds == 0:
            overall_max_hours = 5.0  # Default if all results have 0 time (e.g., 0-5h range)
        else:
            overall_max_hours = max_total_seconds / 3600.0

    # Split base-YOLO vs DCA-YOLO
    yolo_res = [r for r in results if not r.model.startswith("dca_")]
    dca_res  = [r for r in results if     r.model.startswith("dca_")]

    _plot_group(
        results=yolo_res,
        filepath=perf_dir / f"yolo_performance.{file_format}",
        overall_max_hours=overall_max_hours,
    )
    _plot_group(
        results=dca_res,
        filepath=perf_dir / f"dca_yolo_performance.{file_format}",
        type="dca_yolo",
        overall_max_hours=overall_max_hours,
    )
    _save_colour_legend(perf_dir / f"colour_legend.{file_format}")
    _save_marker_legend(perf_dir / f"marker_legend.{file_format}")

    LOGGER.info("All done ✔")


# --------------------------------------------------------------------------- #
# Data collection helpers
# --------------------------------------------------------------------------- #

_MODEL_MARKERS: Dict[ModelSize, str] = {
    "v8s": "D",   # ♦ diamond
    "v8m": "s",   # ■ square
    "v8l": "^",   # ▲ triangle-up
}
_OPTIMISER_COLOURS: Dict[OptimiserName, str] = {
    "cmaes":          "#0077BB",  # blue
    "gpsampler":      "#33BBEE",  # green
    "random":         "#009988",  # grey
    "tpe":            "#EE7733",  # orange
    "ultralytics_es": "#CC3311",  # red
}

# ---- Ultralytics-ES helpers ----------------------------------------------- #

_ITER_LINE = re.compile(r"\(\s*([0-9.]+)s\)")

def _ultra_total_seconds(yaml_f: Path) -> float:
    """Return optimisation wall-clock seconds from best_hyperparameters.yaml."""
    with yaml_f.open("r", encoding="utf-8") as fh:
        header = fh.readline()
    match = _ITER_LINE.search(header)
    if not match:
        raise RuntimeError(f"Cannot parse optimisation time from {yaml_f}")
    return float(match.group(1))


def _ultra_best_f1(ultra_folder: Path) -> float:
    """Reuse `_scan_ultralytics_es` supplied by the user."""
    from ICA_Detection.optimization.analyze.fetch_best_trial import _scan_ultralytics_es
    _, best_f1, *_ = _scan_ultralytics_es(ultra_folder.parent)
    return float(best_f1)

# ---- Optuna helpers -------------------------------------------------------- #

def _optuna_total_seconds(db_path: Path) -> float:
    con = sqlite3.connect(str(db_path))
    st, et = con.execute(
        "SELECT MIN(datetime_start), MAX(datetime_complete) FROM trials"
    ).fetchone()
    con.close()
    return (datetime.fromisoformat(et) - datetime.fromisoformat(st)).total_seconds()


def _optuna_best_f1(db_path: Path) -> float:
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=db_path.stem, storage=storage)
    return float(study.best_value)

# ---- Directory scan -------------------------------------------------------- #

def _collect_all_results(root: Path) -> List[OptimisationResult]:
    """Traverse CADiCA tree and return metrics for every optimiser × model."""
    results: List[OptimisationResult] = []

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        LOGGER.info("Scanning %s …", model_name)

        for opt_dir in model_dir.iterdir():
            if opt_dir.name not in _OPTIMISER_COLOURS:
                continue
            optimiser: OptimiserName = opt_dir.name  # type: ignore[assignment]

            try:
                if optimiser == "ultralytics_es":
                    ultra_sub = opt_dir / "ultralytics_es"
                    secs = _ultra_total_seconds(ultra_sub / "best_hyperparameters.yaml")
                    best_f1 = _ultra_best_f1(ultra_sub)
                else:
                    db_file = next(opt_dir.glob("*.db"))
                    secs = _optuna_total_seconds(db_file)
                    best_f1 = _optuna_best_f1(db_file)
            except (FileNotFoundError, StopIteration, RuntimeError) as exc:
                LOGGER.warning("  ↳ %s: %s – skipped", optimiser, exc)
                continue

            results.append(
                OptimisationResult(model_name, optimiser, secs, best_f1)
            )
            LOGGER.info("  ↳ %s: %.2f h, F1=%.4f",
                        optimiser, secs / 3600.0, best_f1)

    return results


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
def _plot_group(
    *,
    results: Iterable[OptimisationResult],
    filepath: Path,
    type: str = "yolo",
    overall_max_hours: float,
) -> None:
    """Scatter-plot the group and draw step-like connections per optimiser."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlabel("Optimisation wall-clock time [hours]", size=20)

    if type == "dca_yolo": 
        ax.set_ylabel("")
        ax.set_yticklabels("")
        ax.text(-0.07, 1.08, "b.", transform=ax.transAxes, fontsize=20,
                fontweight='bold', va='top')
    else:
        ax.set_ylabel("F1-score", size=20)
        ax.text(-0.07, 1.08, "a.", transform=ax.transAxes, fontsize=20,
                fontweight='bold', va='top')


    # Scatter points first
    for r in results:
        size_tag: ModelSize = r.model.rsplit("yolo", 1)[-1]  # 'v8s' etc.
        ax.scatter(
            r.total_seconds / 3600.0,
            r.best_f1,
            marker=_MODEL_MARKERS[size_tag],
            color=_OPTIMISER_COLOURS[r.optimiser],
            edgecolors="black",
            s=60,
            linewidths=0.6,
            zorder=3,
        )

    # Step connections per optimiser
    for opt in _OPTIMISER_COLOURS:
        pts = sorted(
            (r for r in results if r.optimiser == opt),
            key=lambda r: r.total_seconds,
        )
        if len(pts) < 2:
            continue
        xs: List[float] = []
        ys: List[float] = []
        x_prev = pts[0].total_seconds / 3600.0
        y_prev = pts[0].best_f1
        for nxt in pts[1:]:
            x_next = nxt.total_seconds / 3600.0
            y_next = nxt.best_f1
            # vertical segment
            xs.extend([x_prev, x_prev])
            ys.extend([y_prev, y_next])
            # horizontal segment
            xs.append(x_next)
            ys.append(y_next)
            x_prev, y_prev = x_next, y_next
        ax.plot(xs, ys, color=_OPTIMISER_COLOURS[opt],
                linewidth=1.0, zorder=2, alpha=0.5, linestyle="-")

    # Set consistent X-axis ticks and limits
    upper_xlim_rounded = np.ceil(overall_max_hours / 5.0) * 5.0
    if upper_xlim_rounded == 0: # Ensure at least a 0-5 range if overall_max_hours is very small
        upper_xlim_rounded = 5.0
    
    ax.set_xticks(np.arange(0, upper_xlim_rounded + 1, 5)) # Ticks every 5 hours
    ax.set_xlim(left=-0.5, right=upper_xlim_rounded + 0.5) # Add small padding

    ax.set_yticks(np.arange(0.15, 0.26, 0.02))
    ax.tick_params(axis='both', which='major', labelsize=16) # For label sizes

    # Remove top and right ticks
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    # ax.grid(False) # scienceplots style includes "grid"; uncomment to remove grid
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s", filepath.name)


def _save_colour_legend(filepath: Path) -> None:
    """Save a horizontal, transparent optimiser-colour legend."""
    fig, ax = plt.subplots(figsize=(6, 0.8))
    handles = [
        plt.Line2D([], [], linestyle="none", marker="o",
                   color=col, markersize=8,
                   markeredgecolor="black",
                   label=opt.upper().replace("_ES", "-ES"))
        for opt, col in _OPTIMISER_COLOURS.items()
    ]
    ax.legend(handles=handles, loc="center", ncol=len(handles),
              frameon=False, handlelength=1.5)
    ax.axis("off")
    fig.savefig(filepath, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    LOGGER.info("Saved %s", filepath.name)


def _save_marker_legend(filepath: Path) -> None:
    """Save a horizontal, transparent model-size marker legend."""
    fig, ax = plt.subplots(figsize=(4, 0.8))
    handles = [
        plt.Line2D([], [], linestyle="none",
                   marker=_MODEL_MARKERS[size],
                   markerfacecolor="white",
                   markeredgecolor="black",
                   markersize=8,
                   label=f"v8{size[-1]}")
        for size in ["v8s", "v8m", "v8l"]
    ]
    ax.legend(handles=handles, loc="center", ncol=len(handles),
              frameon=False, handlelength=1.5)
    ax.axis("off")
    fig.savefig(filepath, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    LOGGER.info("Saved %s", filepath.name)


# --------------------------------------------------------------------------- #
# CLI entry-point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Generate CADiCA optimisation plots")
    parser.add_argument("--base-dir", required=True, type=Path,
                        help="Folder with yolov8*/ and dca_yolov8*/")
    parser.add_argument("--out", required=True, type=Path,
                        help="Destination root; <out>/performance/ will be created")
    parser.add_argument("--fmt", default="pdf",
                        help="Output file format (default: pdf)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose console output")
    ns = parser.parse_args()

    if ns.verbose:
        LOGGER.setLevel(logging.DEBUG)

    generate_plots(ns.base_dir, ns.out, ns.fmt)
