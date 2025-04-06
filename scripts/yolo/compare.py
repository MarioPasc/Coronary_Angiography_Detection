import matplotlib.pyplot as plt

import pandas as pd

# Load the two CSV files
path_mga = (
    "/home/mariopasc/Python/Datasets/COMBINED/detection/runs/train/mga_yolo/results.csv"
)
path_non_mga = (
    "/home/mariopasc/Python/Datasets/COMBINED/detection/runs/train/exp/results.csv"
)

df_mga = pd.read_csv(path_mga)
df_non_mga = pd.read_csv(path_non_mga)


# Calculate F1-Score = 2 * (precision * recall) / (precision + recall)
def compute_f1(df):
    precision = df["metrics/precision(B)"]
    recall = df["metrics/recall(B)"]
    return 2 * (precision * recall) / (precision + recall)


# Compute F1 for both models
df_mga["F1-Score"] = compute_f1(df_mga)
df_non_mga["F1-Score"] = compute_f1(df_non_mga)

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
    axs[i].plot(df_mga["epoch"], df_mga[metric], label="YOLO-MGA", linewidth=2)
    axs[i].plot(
        df_non_mga["epoch"],
        df_non_mga[metric],
        label="YOLOv8",
        linewidth=2,
        linestyle="--",
    )
    axs[i].set_title(metric, fontsize=12)
    axs[i].set_xlabel("Epoch")
    axs[i].set_ylabel(metric)
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.savefig(
    "/home/mariopasc/Python/Datasets/COMBINED/detection/runs/train/comparison_metrics.png"
)
plt.show()
