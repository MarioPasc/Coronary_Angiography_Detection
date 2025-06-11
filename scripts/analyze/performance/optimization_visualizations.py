#!/usr/bin/env python3
"""
optimization_visualizations.py
==============================

Generates hyperparameter overview and performance overview plots.
"""

from __future__ import annotations

from pathlib import Path
import sys
import argparse
import logging

# Add the parent directory of ICA_Detection to the Python path
# This is necessary to import modules from ICA_Detection
# Adjust the number of .parent calls if your script is located elsewhere
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3] # Assuming scripts/analyze/performance/ is 3 levels deep from project root
sys.path.append(str(PROJECT_ROOT))

# Import the refactored function from hyperparameter_overview
from ICA_Detection.optimization.analyze.plots.hyperparameter_overview import generate_hyperparameter_figures
from ICA_Detection.optimization.analyze.plots.performance_overview import generate_plots as generate_performance_plots
from ICA_Detection.optimization.analyze.plots.performance_overview import LOGGER as PERFORMANCE_LOGGER
from ICA_Detection.optimization import LOGGER as OPTIMIZATION_LOGGER # Main logger from optimization package
# It's good practice for each script/module to define its own logger if specific formatting is needed
# For hyperparameter_overview, it also uses ICA_Detection.optimization.LOGGER

# Setup a basic logger for this script if not configured by imported modules
VIS_LOGGER = logging.getLogger("optimization_visualizations")
if not VIS_LOGGER.handlers:
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    VIS_LOGGER.addHandler(stream_handler)
VIS_LOGGER.setLevel(logging.INFO)


# ─────────────────────────── CONFIG ───────────────────────────────────── #

# Define global paths - adjust these as per your project structure
DEFAULT_BASE_DIR = Path("/media/mpascual/PortableSSD/Coronariografías/CompBioMed/bho_compbiomed/cadica")
# Sub-paths will be derived from base_dir in the function

# Default values from the original scripts / or sensible defaults
DEFAULT_N_TRIALS_HYPERPARAM = 100
DEFAULT_N_TRIALS_PERFORMANCE = 100
DEFAULT_FILE_FORMAT = "pdf"

# ─────────────────────────────  MAIN  ─────────────────────────────────── #

def run_visualizations(
    base_dir: Path = DEFAULT_BASE_DIR,
    fmt: str = DEFAULT_FILE_FORMAT,
    n_trials_hyperparam: int = DEFAULT_N_TRIALS_HYPERPARAM,
    n_trials_performance: int = DEFAULT_N_TRIALS_PERFORMANCE,
    verbose_performance: bool = False, # Specific verbosity for performance plots
    verbose_hyperparam: bool = False, # Specific verbosity for hyperparam plots (if its logger is used)
) -> None:
    """
    Generates all optimization-related visualizations.
    """
    VIS_LOGGER.info(f"Starting all visualization generation processes.")
    VIS_LOGGER.info(f"Base directory: {base_dir}")
    VIS_LOGGER.info(f"Output format: {fmt}")
    VIS_LOGGER.info(f"Hyperparameter trials: {n_trials_hyperparam}")
    VIS_LOGGER.info(f"Performance trials: {n_trials_performance}")

    # Derive paths from the base_dir
    optimization_dir = base_dir / "optimization"
    kfold_csv_path = base_dir / "kfold" / "CADICA_kfold_metrics.csv"
    gpus_csv_path = base_dir / "gpu_usage_combined" / "gpu_usage_combined.csv"
    # Figures output directory will be managed by the individual plot generation scripts,
    # typically <base_dir>/figures/<plot_type>/

    # Ensure base figures directory exists (though sub-scripts might also do this)
    figures_base_out_dir = base_dir / "figures"
    figures_base_out_dir.mkdir(parents=True, exist_ok=True)


    # --- Generate Hyperparameter Overview Plots ---
    # The hyperparameter_overview script uses ICA_Detection.optimization.LOGGER
    if verbose_hyperparam:
        OPTIMIZATION_LOGGER.setLevel(logging.DEBUG)
        for handler in OPTIMIZATION_LOGGER.handlers: # Apply to all its handlers
            handler.setLevel(logging.DEBUG)
        VIS_LOGGER.info("Verbose logging enabled for hyperparameter plot generation.")
    else: # Ensure it's at INFO if not verbose, in case it was set DEBUG by another call
        OPTIMIZATION_LOGGER.setLevel(logging.INFO)
        for handler in OPTIMIZATION_LOGGER.handlers:
            handler.setLevel(logging.INFO)


    VIS_LOGGER.info("Calling hyperparameter overview plot generation...")
    generate_hyperparameter_figures(
        base_dir=base_dir, # Pass the main base_dir
        output_format=fmt,
        num_trials=n_trials_hyperparam
    )
    VIS_LOGGER.info("Finished hyperparameter overview plot generation call.")


    # --- Generate Performance Overview Plots ---
    if verbose_performance:
        PERFORMANCE_LOGGER.setLevel(logging.DEBUG)
        for handler in PERFORMANCE_LOGGER.handlers: # Apply to all its handlers
            handler.setLevel(logging.DEBUG)
        VIS_LOGGER.info("Verbose logging enabled for performance plot generation.")
    else: # Ensure it's at INFO if not verbose
        PERFORMANCE_LOGGER.setLevel(logging.INFO)
        for handler in PERFORMANCE_LOGGER.handlers:
            handler.setLevel(logging.INFO)


    VIS_LOGGER.info("Calling performance overview plot generation...")
    # performance_overview.generate_plots expects 'plot_root' to be the parent of 'performance' subdir
    # So, figures_base_out_dir is correct here.
    
    # Check for optional files for performance plots
    actual_kfold_csv_path = kfold_csv_path if kfold_csv_path.exists() else None
    actual_gpus_csv_path = gpus_csv_path if gpus_csv_path.exists() else None

    if not actual_kfold_csv_path:
        VIS_LOGGER.warning(f"KFold CSV not found at {kfold_csv_path}, performance plots will not have CV error bars.")
    if not actual_gpus_csv_path:
        VIS_LOGGER.warning(f"GPUs CSV not found at {gpus_csv_path}, performance plots will use default GPU assumptions.")

    generate_performance_plots(
        root=optimization_dir,      # This is <base_dir>/optimization
        plot_root=figures_base_out_dir, # This is <base_dir>/figures
        file_format=fmt,
        cv_csv=actual_kfold_csv_path,
        gpu_csv=actual_gpus_csv_path,
        n_trials=n_trials_performance,
    )
    VIS_LOGGER.info("Finished performance overview plot generation call.")
    VIS_LOGGER.info(f"All plot generation processes initiated. Check respective logs for details.")
    VIS_LOGGER.info(f"Figures should be in subdirectories under: {figures_base_out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate optimization visualization plots.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=f"Root directory for data (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--fmt",
        type=str,
        default=DEFAULT_FILE_FORMAT,
        help=f"Output image format (e.g., pdf, png) (default: {DEFAULT_FILE_FORMAT})",
    )
    parser.add_argument(
        "--n-trials-hyperparam",
        type=int,
        default=DEFAULT_N_TRIALS_HYPERPARAM,
        help=f"Number of trials for hyperparameter plots (default: {DEFAULT_N_TRIALS_HYPERPARAM}).",
    )
    parser.add_argument(
        "--n-trials-performance",
        type=int,
        default=DEFAULT_N_TRIALS_PERFORMANCE,
        help=f"Number of trials for performance plots (default: {DEFAULT_N_TRIALS_PERFORMANCE})",
    )
    parser.add_argument(
        "--verbose-performance",
        action="store_true",
        help="Enable verbose logging for performance plot generation.",
    )
    parser.add_argument(
        "--verbose-hyperparam",
        action="store_true",
        help="Enable verbose logging for hyperparameter plot generation.",
    )
    parser.add_argument(
        "--verbose-all",
        action="store_true",
        help="Enable verbose logging for all plot generation scripts and this orchestrator.",
    )
    args = parser.parse_args()

    if args.verbose_all:
        VIS_LOGGER.setLevel(logging.DEBUG)
        for handler in VIS_LOGGER.handlers: handler.setLevel(logging.DEBUG)
        args.verbose_performance = True # Override if verbose-all is set
        args.verbose_hyperparam = True  # Override if verbose-all is set
        VIS_LOGGER.debug("Verbose logging enabled for orchestrator and overriding sub-script verbosity.")


    run_visualizations(
        base_dir=args.base_dir,
        fmt=args.fmt,
        n_trials_hyperparam=args.n_trials_hyperparam,
        n_trials_performance=args.n_trials_performance,
        verbose_performance=args.verbose_performance,
        verbose_hyperparam=args.verbose_hyperparam,
    )