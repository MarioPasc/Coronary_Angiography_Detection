#!/usr/bin/env python3
"""
generate_latex_table.py

Create a LaTeX table that summarises 3-fold-CV results for YOLOv8 and
DCA-YOLOv8 hyper-parameter optimisation experiments.

  Dataset | Model | Size | CMAES | GPSAMPLER | TPE | RANDOM | ULTRALYTICS ES
Each cell shows mean ± error (half-width of a two-sided (1–alpha) CI).  
The best optimiser per (Dataset, Model, Size) row is typeset in **bold**.  
Repeated Dataset / Model entries are blanked to avoid duplication.

"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats # type: ignore[import-untyped]


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

# ---------------------------------------------------------------------------
# REPLACE the old 'combine_tables' and 'to_latex' with the versions below
# ---------------------------------------------------------------------------

def combine_tables(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Return a tidy DataFrame ready for LaTeX with layout:

        Optimiser | Size |  (Dataset₁,Model₁) … (Datasetₙ,Model₂)

    i.e. three rows per optimiser (sizes v8s/m/l) and
    two columns per dataset (DCA-YOLOv8, YOLOv8).
    """
    if not frames:
        return pd.DataFrame()                      # nothing to do

    long = pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ #
    # Make "μ ± Δ" strings **and** remember numeric μ for “best optimizer”
    # ------------------------------------------------------------------ #
    long["stat"] = long.apply(lambda r: f"{r['μ']:.3f} ± {r['Δ']:.3f}", axis=1)
    long["__mu__"] = long["μ"]                  # keep plain mean for ranking

    # Mark best optimiser (max μ) for every (Dataset,Model,Size)
    best_mask = (
        long
        .groupby(["Dataset", "Model", "Size"])["__mu__"]
        .transform(lambda s: s == s.max())
    )
    long.loc[best_mask, "stat"] = (
        r"\textbf{" + long.loc[best_mask, "stat"] + r"}"
    )

    # ------------------------------------------------------------------ #
    # Pivot so rows = (Optimiser, Size)   and   columns = (Dataset, Model)
    # ------------------------------------------------------------------ #
    wide = long.pivot(
        index=["Optimiser", "Size"],
        columns=["Dataset", "Model"],
        values="stat",
    ).sort_index(axis=1, level=[0, 1])           # keep datasets alphabetic

    # Bring optimiser order back to the one defined in OPTIMISER_ORDER
    opt_order = [k for k, _ in OPTIMISER_ORDER]
    wide = wide.reindex(opt_order, level=0)      # row-level re-order

    # Compact index for nicer LaTeX: Optimiser shown once every three rows
    wide = wide.reset_index()
    wide["Optimiser"] = pd.Categorical(
        wide["Optimiser"], categories=opt_order, ordered=True
    )
    return wide


def to_latex(df: pd.DataFrame, ci_value: float) -> str:
    """
    Render our wide table with a two-line header:

         ┌───────────── top line  ──────────────┐
         │   Dataset₁        …      Datasetₙ    │
         │ DCA-YOLOv8 YOLOv8 … DCA-YOLOv8 YOLOv8│
         └──────────────────────────────────────┘
    """
    if df.empty:
        return "% (no data)\n"

    # ----- 1. Build column format ------------------------------------------------
    n_datasets = (df.columns.nlevels == 3 and df.columns.get_level_values(0)[3:].nunique()) or \
                 (df.columns.nlevels == 2 and df.columns.get_level_values(0)[3:].nunique())
    # 3 leading cols (Optimiser, Size) + 2 per dataset
    col_fmt = "ll" + "c" * (df.shape[1] - 2)

    # ----- 2. Flatten the MultiIndex columns to strings Pandas can keep ----------
    #        We'll supply real multicolumn commands ourselves.
    flat_cols = [
        "{}␟{}".format(*c) if isinstance(c, tuple) else c  # ␟ is just a placeholder
        for c in df.columns
    ]
    out = df.set_axis(flat_cols, axis=1)

    # ----- 3. Dump to latex WITHOUT header and index -----------------------------
    body = out.to_latex(index=False,
                        header=False,
                        escape=False,
                        column_format=col_fmt,
                        multicolumn=False)

    # ----- 4. Craft a two-row header by hand -------------------------------------
    # split again on our placeholder
    header_groups = [
        tuple(col.split("␟")) if "␟" in col else (col,)
        for col in flat_cols
    ]

    top = []
    bottom = []
    span = 0
    for h in header_groups:
        if len(h) == 1:          # "Optimiser" or "Size"
            if span:
                top.append(f"& \\multicolumn{{{span}}}{{c}}{{}}")
                span = 0
            top.append(f"& {h[0]}")
            bottom.append(f"& {h[0]}")
        else:                    # (Dataset, Model)
            ds, mdl = h
            if span == 0:
                top.append("& \\multicolumn{{2}}{{c}}{{" + ds + "}}")
            span = (span + 1) % 2
            bottom.append(f"& {mdl}")
    if span:                     # flush remainder
        top.append(f"& \\multicolumn{{{span}}}{{c}}{{}}")

    top_line = " ".join(top).lstrip("& ") + r"\\"
    mid_line = " ".join(bottom).lstrip("& ") + r"\\"
    midrule = r"\midrule"
    toprule = r"\toprule"

    header = (
        toprule + "\n"
        + top_line + "\n"
        + r"\cmidrule(lr){3-" + f"{df.shape[1]}" + r"}" + "\n"
        + mid_line + "\n"
        + midrule + "\n"
    )

    # ----- 5. Splice header + body and wrap with \begin{center} ------------------
    table = header + "\n".join(body.splitlines()[3:])  # drop Pandas’ own toprule etc.

    caption = (
        "Mean $F_1$-score at 0.5 IoU under 3-fold CV "
        f"($\\pm$ {int(ci_value*100)}\\,\\% CI)."
    )

    return (
        "\\begin{center}\n"
        "\\footnotesize\n"
        "\\begin{tabular}{" + col_fmt + "}\n"
        + table +
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        "\\label{{tab:hpo_results}}\n"
        "\\end{center}\n"
    )



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
