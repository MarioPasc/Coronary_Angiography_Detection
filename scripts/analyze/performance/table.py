#!/usr/bin/env python3
"""
generate_latex_table.py

Create a LaTeX table that summarises 3-fold-CV results for YOLOv8 and
DCA-YOLOv8 hyper-parameter optimisation experiments.

Columns:
  Dataset | Model | Size | CMAES | GPSAMPLER | TPE | RANDOM | ULTRALYTICS ES
Each cell shows mean ± error (half-width of a two-sided (1–alpha) CI).  
The best optimiser per (Dataset, Model, Size) row is typeset in **bold**.  
Repeated Dataset / Model entries are blanked to avoid duplication.

Author: Your Name <you@example.com>
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


# ---------- configuration -------------------------------------------------- #

OPTIMISER_ORDER = [
    ("cmaes", "CMAES"),
    ("gpsampler", "GPSAMPLER"),
    ("tpe", "TPE"),
    ("random", "RANDOM"),
    ("ultralytics_es", "ULTRALYTICS ES"),
]

DEFAULT_METRIC = "f1_score"
DEFAULT_CI = 0.95


# ---------- helper utilities ---------------------------------------------- #

def ci_halfwidth(series: pd.Series, alpha: float = DEFAULT_CI) -> float:
    """Return half-width of a two-sided alpha-level t-CI for the mean."""
    n = series.notna().sum()
    if n < 2:
        return np.nan
    mean = series.mean()
    sem = stats.sem(series, nan_policy="omit")
    df = n - 1
    t_crit = stats.t.ppf((1 + alpha) / 2, df)
    return t_crit * sem


@dataclass(frozen=True, slots=True)
class Variant:
    model: str  # "YOLOv8" or "DCA-YOLOv8"
    size: str   # "v8s" | "v8m" | "v8l"

    @classmethod
    def parse(cls, text: str) -> "Variant":
        """
        Examples
        --------
        yolov8s           -> (YOLOv8, v8s)
        dca_yolov8m       -> (DCA-YOLOv8, v8m)
        """
        if text.startswith("dca_"):
            model = "DCA-YOLOv8"
            core = text.removeprefix("dca_")
        else:
            model = "YOLOv8"
            core = text
        m = re.fullmatch(r"yolov8([sml])", core)
        if not m:
            raise ValueError(f"Unrecognised variant: '{text}'")
        size = f"v8{m.group(1)}"
        return cls(model, size)


def load_metrics(path: Path, metric: str, alpha: float) -> pd.DataFrame:
    """Read a *_kfold_metrics.csv file and return summary statistics."""
    df = pd.read_csv(path, low_memory=False)

    # Derive logical fields that are not present in the CSV
    dataset = path.stem.split("_")[0]
    variants = df["model_variant"].map(Variant.parse)
    df[["Model", "Size"]] = pd.DataFrame(
        [(v.model, v.size) for v in variants], index=df.index
    )
    df["Dataset"] = dataset
    df["Optimiser"] = df["optimizer"].str.lower()

    # Aggregate across folds
    grouped = (
        df.groupby(["Dataset", "Model", "Size", "Optimiser"], sort=False)[metric]
        .agg(["mean", lambda s: ci_halfwidth(s, alpha=alpha)])
        .rename(columns={"mean": "μ", "<lambda_0>": "Δ"})
        .reset_index()
    )
    return grouped


def combine_tables(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-CSV summaries and pivot to the desired wide format."""
    wide_frames: list[pd.DataFrame] = []
    for g in frames:
        # Build "mean ± CI" string
        g["stat"] = g.apply(
            lambda r: f"{r['μ']:.3f} ± {r['Δ']:.3f}", axis=1
        )
        # Pivot so each optimiser becomes a column
        pivot = (
            g.pivot(
                index=["Dataset", "Model", "Size"],
                columns="Optimiser",
                values="stat",
            )
            .reindex([k for k, _ in OPTIMISER_ORDER], axis=1)
        )
        wide_frames.append(pivot)

    if not wide_frames:
        # Return an empty DataFrame with expected columns if no data
        empty_cols = ["Dataset", "Model", "Size"] + [label for _, label in OPTIMISER_ORDER]
        return pd.DataFrame(columns=empty_cols)

    merged = pd.concat(wide_frames).sort_index()

    # Highlight the best (highest μ) in bold
    def bold_best(row: pd.Series) -> pd.Series:
        # Extract the numeric means for comparison
        means = row.str.extract(r"^([0-9.]+)", expand=False).astype(float)
        best = means.idxmax()
        if pd.notna(best) and best in row and pd.notna(row[best]):
            row[best] = rf"\textbf{{{row[best]}}}"
        return row

    merged = merged.apply(bold_best, axis=1)
    merged = merged.fillna("")

    # Reset index to operate on columns for multirow
    df_for_latex = merged.reset_index()

    if df_for_latex.empty:
        return df_for_latex # Return empty if no data after processing

    # Apply multirow for 'Dataset'
    new_dataset_col = df_for_latex['Dataset'].copy()
    if not df_for_latex.empty:
        current_dataset_val = df_for_latex.loc[0, 'Dataset']
        current_dataset_start_idx = 0
        for i in range(1, len(df_for_latex)):
            if df_for_latex.loc[i, 'Dataset'] != current_dataset_val:
                count = i - current_dataset_start_idx
                if count > 1:
                    new_dataset_col[current_dataset_start_idx] = \
                        rf"\multirow{{{count}}}{{*}}{{{current_dataset_val}}}"
                    for j in range(current_dataset_start_idx + 1, i):
                        new_dataset_col[j] = ""
                current_dataset_val = df_for_latex.loc[i, 'Dataset']
                current_dataset_start_idx = i
        # Process the last dataset block
        count = len(df_for_latex) - current_dataset_start_idx
        if count > 1:
            new_dataset_col[current_dataset_start_idx] = \
                rf"\multirow{{{count}}}{{*}}{{{current_dataset_val}}}"
            for j in range(current_dataset_start_idx + 1, len(df_for_latex)):
                new_dataset_col[j] = ""
        df_for_latex['Dataset'] = new_dataset_col

    # Apply multirow for 'Model' (within each Dataset group)
    new_model_col = df_for_latex['Model'].copy()
    if not df_for_latex.empty:
        current_model_val = df_for_latex.loc[0, 'Model']
        # Use original dataset value from the sorted MultiIndex for correct model grouping
        current_dataset_group_val_for_model = merged.index[0][0] 
        current_model_start_idx = 0

        for i in range(1, len(df_for_latex)):
            original_dataset_at_i = merged.index[i][0] # Dataset from original multi-index
            if (df_for_latex.loc[i, 'Model'] != current_model_val or
                original_dataset_at_i != current_dataset_group_val_for_model):
                count = i - current_model_start_idx
                if count > 1:
                    new_model_col[current_model_start_idx] = \
                        rf"\multirow{{{count}}}{{*}}{{{current_model_val}}}"
                    for j in range(current_model_start_idx + 1, i):
                        new_model_col[j] = ""
                
                current_model_val = df_for_latex.loc[i, 'Model']
                current_dataset_group_val_for_model = original_dataset_at_i
                current_model_start_idx = i
        
        # Process the last model block
        count = len(df_for_latex) - current_model_start_idx
        if count > 1:
            new_model_col[current_model_start_idx] = \
                rf"\multirow{{{count}}}{{*}}{{{current_model_val}}}"
            for j in range(current_model_start_idx + 1, len(df_for_latex)):
                new_model_col[j] = ""
        df_for_latex['Model'] = new_model_col
    
    return df_for_latex


def to_latex(df: pd.DataFrame, ci_value: float) -> str:
    """Render DataFrame to LaTeX using booktabs + multirow."""
    # Get mappings of lowercase keys to uppercase labels
    optimiser_map = {k: label for k, label in OPTIMISER_ORDER}
    
    # Check which optimiser columns are actually present in the DataFrame
    # Note: The column names in df are lowercase after pivoting
    present_optimiser_keys = []
    for k, _ in OPTIMISER_ORDER:
        if k in df.columns:
            present_optimiser_keys.append(k)
    
    # Define column order with actual present columns (using lowercase keys)
    column_order = ["Dataset", "Model", "Size"] + present_optimiser_keys
    
    # Handle cases where df might be empty or not have all expected columns
    if df.empty:
        df = pd.DataFrame(columns=column_order)
    else:
        # Make sure we're not dropping any columns - only reordering them
        # Keep only columns that exist in df to avoid KeyError
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
    
    # Calculate the proper column format string based on actual columns
    col_format = "lll" + "c" * len(present_optimiser_keys)
    
    # Use the uppercase labels for the header but lowercase keys for accessing columns
    header = [
        "Dataset",
        "Model",
        "Size",
        *[optimiser_map[k] for k in present_optimiser_keys]
    ]
    
    latex = df.to_latex(
        index=False,
        escape=False,
        multicolumn=False,
        multicolumn_format="c",
        column_format=col_format,
        caption=(
            "Mean $F_1$‐score at 0.5 IoU under 3-fold cross-validation "
            f"($\\pm$ {int(ci_value*100)} \\% CI) for each optimiser."
        ),
        label="tab:cadica_hpo_results",
        longtable=False,
        bold_rows=False,
        na_rep="",
        header=header,
    )
    # Optional: centre the table
    return "\\begin{center}\n" + latex + "\\end{center}\n"


# ---------- CLI ------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate LaTeX summary table for YOLOv8 HPO experiments."
    )
    p.add_argument(
        "csvs",
        nargs="+",
        type=Path,
        help="One or more *_kfold_metrics.csv files.",
    )
    p.add_argument(
        "-m",
        "--metric",
        default=DEFAULT_METRIC,
        help=f"Metric column to use (default: {DEFAULT_METRIC}).",
    )
    p.add_argument(
        "--ci",
        type=float,
        default=DEFAULT_CI,
        help="Confidence level for the CI (default: 0.95).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default="/media/mpascual/PortableSSD/Coronariografías/CompBioMed/bho_compbiomed/cadica/figures/table/table.tex",
        help="Write LaTeX to this file instead of stdout.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    frames = [load_metrics(p, args.metric, args.ci) for p in args.csvs]
    table = combine_tables(frames)
    latex = to_latex(table, args.ci)

    if args.output:
        args.output.write_text(latex)
        print(f"[+] LaTeX written to {args.output}")
    else:
        print(latex)


if __name__ == "__main__":
    main()
