import matplotlib.pyplot as plt
import pandas as pd  # type: ignore

# Parameters dictionary - define models and their paths
params = {
    "models": {
        "YOLOv8": "/home/mariopasc/Python/Datasets/COMBINED/detection/runs/train/base/results.csv",
        "MGA-CBAM ADD-MULT": "/home/mariopasc/Python/Datasets/COMBINED/detection/runs/train/mga_cbam_yolo_samcamADD_pyramidMULTIPLY/results.csv",
        "MGA-CBAM ADD-ADD": "/home/mariopasc/Python/Datasets/COMBINED/detection/runs/train/mga_cbam_yolo_samcamADD_pyramidADD/results.csv",
    },
    "output_path": "/home/mariopasc/Python/Datasets/COMBINED/detection/runs/train/comparison_metrics.png",
}


# Calculate F1-Score = 2 * (precision * recall) / (precision + recall)
def compute_f1(df):
    precision = df["metrics/precision(B)"]
    recall = df["metrics/recall(B)"]
    return 2 * (precision * recall) / (precision + recall)


# Load all dataframes and compute F1 scores
dfs = {}
for model_name, path in params["models"].items():  # type: ignore
    df = pd.read_csv(path)
    df["F1-Score"] = compute_f1(df)
    dfs[model_name] = df

# Define metrics to compare
metrics = [
    "metrics/recall(B)",
    "metrics/precision(B)",
    "F1-Score",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "time",
]

# Plot 3x3 comparison
fig, axs = plt.subplots(3, 3, figsize=(18, 12))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    for model_name, df in dfs.items():
        # Use dashed line for YOLOv8 to maintain original styling
        line_style = "--" if model_name == "YOLOv8" else "-"
        axs[i].plot(
            df["epoch"], df[metric], label=model_name, linewidth=2, linestyle=line_style
        )

    axs[i].set_title(metric, fontsize=12)
    axs[i].set_xlabel("Epoch")
    axs[i].set_ylabel(metric)
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.savefig(params["output_path"])
plt.show()
