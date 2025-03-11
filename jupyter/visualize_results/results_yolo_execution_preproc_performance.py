"""
plot_comparison_boxplots.py

Creates a figure with three subplots comparing F1-Score, mAP@50, and mAP@50-95
using boxplots. Each subplot shows boxplots for CADICA, ARCADE, and COMBINED,
where each dataset has two boxes: one for the RAW preprocessing and one for the
CLAHE+FSE preprocessing. Results from two different seeds (two CSV files) are
combined in the final plot.

FUNCTIONS:
    load_and_combine_data(csv_paths: list[str]) -> pd.DataFrame
        Reads multiple CSV files, concatenates them, and parses out dataset and
        preprocessing columns from experiment_name.
    plot_boxplots(df: pd.DataFrame) -> None
        Plots three subplots of horizontal boxplots for F1, mAP@50, and mAP@50-95.
"""

import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt


def load_and_combine_data(csv_paths):
    """
    Reads multiple CSV files, concatenates them, and extracts dataset and
    preprocessing type from the 'experiment_name' column. Returns a single
    pandas DataFrame containing all rows from the CSVs.

    Parameters
    ----------
    csv_paths : list[str]
        List of paths to the CSV files to be read and combined.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing data from all input CSV files, with
        additional 'dataset' and 'preprocessing' columns.
    """
    # Read and store all CSVs in a list
    dfs = []
    for path in csv_paths:
        df_temp = pd.read_csv(path)
        dfs.append(df_temp)

    # Combine into a single DataFrame
    df = pd.concat(dfs, ignore_index=True)

    # Parse out dataset and preprocessing
    # Example of experiment_name: 'CADICA_RAW' or 'CADICAARCADE_CLAHEFSE'
    def parse_experiment_name(name):
        # e.g. name = "CADICAARCADE_CLAHEFSE"
        if "CADICAARCADE" in name:
            dataset_part = "COMBINED"
        elif "CADICA" in name:
            dataset_part = "CADICA"
        elif "ARCADE" in name:
            dataset_part = "ARCADE"
        else:
            dataset_part = "UNKNOWN"  # fallback, should not happen in ideal data

        if "RAW" in name:
            prep_part = "RAW"
        else:
            prep_part = "CLAHEFSE"

        return dataset_part, prep_part

    df["dataset"], df["preprocessing"] = zip(
        *df["experiment_name"].apply(parse_experiment_name)
    )

    return df


def plot_boxplots(df):
    """
    Plots three subplots of horizontal boxplots for F1, mAP@50, and mAP@50-95
    using Seaborn. Each subplot has CADICA, ARCADE, and COMBINED along the
    y-axis with two boxes (RAW and CLAHEFSE).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that must contain the columns:
        ['dataset', 'preprocessing', 'f1', 'map50', 'map50_95'].

    Returns
    -------
    None
        Displays the final figure with three subplots.
    """
    # Set a consistent style
    sns.set_style("whitegrid")

    # Prepare figure with 3 subplots, horizontally aligned
    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(18, 6), sharey=True, sharex=False
    )

    metrics = [("f1", "F1-Score"), ("map50", "mAP@50"), ("map50_95", "mAP@50-95")]
    preprocessing = [("RAW", "RAW"), ("CLAHEFSE", "CLAHE+FSE")]

    for ax, (col, title) in zip(axes, metrics):
        # Create horizontal boxplot
        sns.boxplot(
            ax=ax,
            x=col,  # The metric we want on the x-axis
            y="dataset",  # The dataset categories on the y-axis
            hue="preprocessing",
            data=df,
            orient="h",  # Tells Seaborn it's a horizontal plot
            width=0.6,  # Box width
            showfliers=False,
        )
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("")  # We only need to label the y-axis once (sharey=True)

        # Move the legend out or keep it inside
        ax.get_legend().remove()

    # Create one common legend on the right side
    handles, labels = axes[0].get_legend_handles_labels()
    labels = ["RAW", "CLAHE + FSE"]
    fig.legend(handles, labels, loc="upper right", title="Preprocessing")

    plt.tight_layout()
    plt.savefig("preprocessing_comparison.svg")
    plt.show()


if __name__ == "__main__":
    # Example usage:
    # 1. Load and combine the data from two CSVs
    import os

    base = "/home/mariopasc/x2go_shared"
    name = "preprocessing_performance_results_seed"
    csv_paths = [
        os.path.join(base, name + "42" + ".csv"),
        os.path.join(base, name + "123" + ".csv"),
    ]
    combined_df = load_and_combine_data(csv_paths)

    # 2. Plot the boxplots
    plot_boxplots(combined_df)
