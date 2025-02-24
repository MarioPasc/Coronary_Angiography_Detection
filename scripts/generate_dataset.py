import os
import json

from typing import Dict, Any

from ICA_Detection.generator.generator import DatasetGenerator
from ICA_Detection.splits.holdout import holdout_split  # type: ignore

import logging
import os
from pathlib import Path
import shutil

# ==========================
# = USER-DEFINED VARIABLES =
# ==========================

# Define log file path
LOG_FILE = "./dataset_generation.log"  # Change this to your desired log file path

# Datasets to integrate and combined. Possible args incluide ["CADICA", "ARCADE", "KEMEROVO"]
DATASETS_TO_PROCESS = ["ARCADE"]

# Splits dictionary. Split type and % of the images being allocated in that split
SPLITS_DICT = {"train": 0.7, "val": 0.3, "test": 0.0}

# Output folder to store the final combined, preprocessed dataset and splits
OUTPUT_FOLDER = "/media/hddb/mario/data/COMBINED"
OUTPUT_FOLDER = "/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets"
OUTPUT_FOLDER = "/home/mario/Python/Datasets/COMBINED"


# Preprocessing steps to be performed on the datasets
PLAN_STEPS = {
    "resolution_standarization": {
        "desired_X": 512,
        "desired_Y": 512,
        "method": "bilinear",
    },
    "dtype_standarization": {"desired_dtype": "uint8"},
    "format_standarization": {"desired_format": "png"},
    "clahe": {
        "window_size": 5,
        "sigma": 1.0,
        "clipLimit": 3.0,
        "tileGridSize": (8, 8),
    },
    "filtering_smoothing_equalization": {"window_size": 5, "sigma": 1.0},
    "labels_formats": {"YOLO": True},
}

# ==========================
# ========= SCRIPT =========
# ==========================

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE, mode="a")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# --- Integration Step ---
# You can download the datasets manually from:
# - KEMEROV: https://data.mendeley.com/datasets/ydrm75xywg/2
# - ARCADE: https://zenodo.org/records/10390295
# - CADICA: https://data.mendeley.com/datasets/p9bpx9ctcv/2

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
output_combined_json = os.path.join(OUTPUT_FOLDER, "combined_standardized.json")
output_planned_json = os.path.join(OUTPUT_FOLDER, "planned_standardized.json")

root_dirs = {
    "CADICA": OUTPUT_FOLDER,
    "ARCADE": OUTPUT_FOLDER,
    "KEMEROVO": OUTPUT_FOLDER,
}

arcade_task = "stenosis"

logger.info("Integrating datasets...")
final_json: Dict[str, Any] = DatasetGenerator.integrate_datasets(
    DATASETS_TO_PROCESS, root_dirs, arcade_task=arcade_task
)
with open(output_combined_json, "w") as f:
    json.dump(final_json, f, indent=4)
logger.info(f"Combined standardized JSON saved to {output_combined_json}")

# --- Preprocessing Planning Step ---
with open(output_combined_json, "r") as f:
    data = json.load(f)

logger.info("Creating preprocessing plan...")
planned_data = DatasetGenerator.create_preprocessing_plan(data, PLAN_STEPS)
with open(output_planned_json, "w") as f:
    json.dump(planned_data, f, indent=4)
logger.info(f"Preprocessing plan saved to {output_planned_json}")

# --- Preprocessing Execution Step ---
# Here, the user can supply a list of steps in order.
steps_order = list(PLAN_STEPS.keys())

output_ica_detection = os.path.join(OUTPUT_FOLDER, "ICA_DETECTION")
logger.info("Applying preprocessing plan...")
DatasetGenerator.apply_preprocessing_plan(
    output_planned_json, output_ica_detection, steps_order
)
logger.info("Preprocessing completed.")


def rename_labels_folder(base_output_dir: str) -> None:
    """
    After applying the preprocessing plan, the output folder typically contains:
        base_output_dir/
            images/
            labels/        <- old labeling approach
            labels_yolo/   <- YOLO labels
    We need to remove the 'labels/' folder and rename 'labels_yolo/' to 'labels/'.
    """
    old_labels = Path(base_output_dir) / "labels"
    yolo_labels = Path(base_output_dir) / "labels_yolo"
    if old_labels.exists() and old_labels.is_dir():
        logger.info(f"Removing old labels folder: {old_labels}")
        shutil.rmtree(old_labels)
    if yolo_labels.exists() and yolo_labels.is_dir():
        new_labels = Path(base_output_dir) / "labels"
        logger.info(f"Renaming {yolo_labels} to {new_labels}")
        yolo_labels.rename(new_labels)


rename_labels_folder(output_ica_detection)

shutil.move(src=os.path.join(output_ica_detection, "processed.json"),
            dst=os.path.join(output_ica_detection, ".."))

output_yolo_dataset = os.path.join(OUTPUT_FOLDER, "YOLO_ICA_DETECTION")
yaml_filename = "yolo_ica_detection.yaml"

holdout_split(
    output_ica_detection,
    SPLITS_DICT,
    output_yolo_dataset,
    yaml_filename=yaml_filename,
    splits_info_filename="splits_info.json",
    include_datasets=["ARCADE"],
)
