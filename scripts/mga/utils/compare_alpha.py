import matplotlib.pyplot as plt
import pandas as pd  # type: ignore
import os
import numpy as np
from pathlib import Path
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare YOLOv8 with MGA-YOLO alpha experiments"
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default="/home/mariopasc/Python/Datasets/COMBINED/detection/runs/train/alpha_sweep/alpha_sweep_20250407_161714",
        help="Directory containing alpha sweep experiments",
    )
    parser.add_argument(
        "--yolo-path",
        type=str,
        default="/home/mariopasc/Python/Datasets/COMBINED/detection/runs/train/exp/results.csv",
        help="Path to base YOLOv8 results.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/mariopasc/Python/Datasets/COMBINED/detection/runs/train/alpha_sweep_comparison.png",
        help="Output path for comparison image",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "metrics/recall(B)",
            "metrics/precision(B)",
            "F1-Score",
            "val/box_loss",
            "val/cls_loss",
            "val/dfl_loss",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
        ],
        help="Metrics to compare",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=None,  # Changed default to None - will auto-detect all alpha folders
        help="Specific alpha values to include (default: auto-detect all)",
    )

    return parser.parse_args()


def find_all_alpha_folders(sweep_dir):
    """
    Find all alpha experiment folders and extract their alpha values.

    Args:
        sweep_dir: Directory containing alpha sweep experiments

    Returns:
        List of alpha values found in folder names
    """
    sweep_dir = Path(sweep_dir)
    alpha_values = []

    # Find all folders matching the pattern
    for exp_folder in sweep_dir.glob("mga_yolo_alpha_*"):
        try:
            # Extract alpha value from folder name
            alpha_str = exp_folder.name.split("_")[-1]
            alpha = float(alpha_str)
            alpha_values.append(alpha)
        except (ValueError, IndexError):
            print(
                f"Warning: Could not extract alpha value from folder: {exp_folder.name}"
            )

    # Sort alpha values
    alpha_values.sort()

    return alpha_values


def load_experiment_data(sweep_dir, yolo_path, selected_alphas=None):
    """
    Load data from YOLOv8 and all alpha experiment folders.

    Args:
        sweep_dir: Directory containing alpha sweep experiments
        yolo_path: Path to base YOLOv8 results.csv
        selected_alphas: List of specific alpha values to include (None = all)

    Returns:
        Dictionary of dataframes with experiment names as keys
    """
    # Load base YOLOv8 results
    df_data = {"YOLOv8": pd.read_csv(yolo_path)}

    # Load MGA results for each alpha value
    sweep_dir = Path(sweep_dir)
    for exp_folder in sorted(sweep_dir.glob("mga_yolo_alpha_*")):
        alpha_str = exp_folder.name.split("_")[-1]
        try:
            alpha = float(alpha_str)

            # Skip if not in selected alphas (if provided)
            if selected_alphas is not None and alpha not in selected_alphas:
                continue

            results_path = exp_folder / "results.csv"
            if results_path.exists():
                df = pd.read_csv(results_path)
                # Add F1-Score
                precision = df["metrics/precision(B)"]
                recall = df["metrics/recall(B)"]
                df["F1-Score"] = 2 * (precision * recall) / (precision + recall)

                # Store with alpha value in name
                df_data[f"MGA α={alpha:.2f}"] = df
        except (ValueError, FileNotFoundError) as e:
            print(f"Error processing {exp_folder}: {e}")

    return df_data


def plot_metric_comparison(df_data, metrics, output_path):
    """
    Create comparison plots for each metric.

    Args:
        df_data: Dictionary of dataframes with experiment names as keys
        metrics: List of metrics to plot
        output_path: Path to save the output image
    """
    # Calculate number of rows and columns for subplots
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    # Create a figure with subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    if n_metrics == 1:  # Handle single plot case
        axs = np.array([axs])
    axs = axs.flatten()

    # Create a color gradient for MGA models
    cmap = plt.cm.viridis
    baseline_color = "red"

    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axs[i]

        # Plot YOLOv8 baseline
        yolo_df = df_data["YOLOv8"]
        ax.plot(
            yolo_df["epoch"],
            yolo_df[metric],
            color=baseline_color,
            linewidth=2.5,
            linestyle="-",
            label="YOLOv8",
        )

        # Plot MGA models with different alpha values
        mga_keys = [k for k in df_data.keys() if k != "YOLOv8"]
        mga_alphas = [float(k.split("=")[1]) for k in mga_keys]

        # Sort by alpha value
        sorted_indices = np.argsort(mga_alphas)
        sorted_keys = [mga_keys[i] for i in sorted_indices]
        norm_alphas = [(float(k.split("=")[1]) / max(mga_alphas)) for k in sorted_keys]

        for j, key in enumerate(sorted_keys):
            df = df_data[key]
            color = cmap(norm_alphas[j])
            ax.plot(
                df["epoch"],
                df[metric],
                color=color,
                linewidth=1.5,
                alpha=0.8,
                label=key,
            )

        # Add labels and grid
        ax.set_title(metric, fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)

        # Add legend to the first plot only (to avoid clutter)
        if i == 0:
            ax.legend(fontsize=10, loc="best")

    # Hide any unused subplots
    for i in range(n_metrics, len(axs)):
        axs[i].axis("off")

    # Add a common legend at the bottom
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, len(df_data)),
        bbox_to_anchor=(0.5, 0),
        fontsize=12,
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08 + 0.02 * min(5, len(df_data)))

    # Add title
    fig.suptitle("YOLOv8 vs MGA-YOLO with Different Alpha Values", fontsize=16, y=0.99)

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Comparison saved to {output_path}")

    return fig


def compute_best_alpha_per_metric(df_data, metrics):
    """
    Analyze which alpha value performs best for each metric.

    Args:
        df_data: Dictionary of dataframes with experiment names as keys
        metrics: List of metrics to evaluate

    Returns:
        DataFrame with best alpha values per metric
    """
    results = []

    for metric in metrics:
        metric_results = {}

        # Determine if this is a metric to maximize or minimize
        minimize = metric.startswith(("val/", "train/")) or "loss" in metric.lower()

        for name, df in df_data.items():
            if name == "YOLOv8":
                baseline_value = df[metric].iloc[-1]  # Last epoch value
                metric_results["Baseline"] = baseline_value
            else:
                # Extract alpha value
                alpha = float(name.split("=")[1])
                # Get final value of the metric
                final_value = df[metric].iloc[-1]
                metric_results[alpha] = final_value

        # Find best alpha (excluding baseline)
        alphas = [k for k in metric_results.keys() if k != "Baseline"]
        if alphas:
            if minimize:
                best_alpha = min(alphas, key=lambda a: metric_results[a])
                best_value = metric_results[best_alpha]
                comparison = (
                    "better" if best_value < metric_results["Baseline"] else "worse"
                )
            else:
                best_alpha = max(alphas, key=lambda a: metric_results[a])
                best_value = metric_results[best_alpha]
                comparison = (
                    "better" if best_value > metric_results["Baseline"] else "worse"
                )

            results.append(
                {
                    "Metric": metric,
                    "Best Alpha": best_alpha,
                    "Best Value": best_value,
                    "Baseline Value": metric_results["Baseline"],
                    "Comparison": comparison,
                }
            )

    return pd.DataFrame(results)


def main():
    args = parse_arguments()

    # Detect all alpha folders if no specific alphas are provided
    if args.alphas is None:
        print(f"Auto-detecting alpha values in {args.sweep_dir}...")
        all_alphas = find_all_alpha_folders(args.sweep_dir)
        print(f"Found {len(all_alphas)} alpha values: {all_alphas}")
        args.alphas = all_alphas

    # Load data from all experiments
    print(f"Loading experiment data from {args.sweep_dir}")
    df_data = load_experiment_data(args.sweep_dir, args.yolo_path, args.alphas)

    # Add F1-score for YOLOv8
    yolo_df = df_data["YOLOv8"]
    precision = yolo_df["metrics/precision(B)"]
    recall = yolo_df["metrics/recall(B)"]
    yolo_df["F1-Score"] = 2 * (precision * recall) / (precision + recall)

    # Plot comparisons
    print(f"Creating comparison plots for {len(df_data)} models")
    fig = plot_metric_comparison(df_data, args.metrics, args.output)

    # Find best alpha per metric
    best_alpha_df = compute_best_alpha_per_metric(df_data, args.metrics)

    # Print best alpha analysis
    print("\nBest Alpha Values Per Metric:")
    print("=" * 80)
    print(best_alpha_df.to_string(index=False))
    print("=" * 80)

    # Save best alpha analysis
    output_dir = os.path.dirname(args.output)
    analysis_path = os.path.join(output_dir, "alpha_analysis.csv")
    best_alpha_df.to_csv(analysis_path, index=False)
    print(f"Alpha analysis saved to {analysis_path}")

    plt.show()


if __name__ == "__main__":
    main()
