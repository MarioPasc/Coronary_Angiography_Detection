import os
import json

from typing import Dict, Any

from ICA_Detection.generator.generator import DatasetGenerator

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
DATASETS_TO_PROCESS = ["CADICA"]

# Splits dictionary. Split type and % of the images being allocated in that split
SPLITS_DICT = {"train": 0.7, "val": 0.3, "test": 0.0}

# Output folder to store the final combined, preprocessed dataset and splits
# OUTPUT_FOLDER = "/home/mario/Python/Datasets/COMBINED" # Portátil
# OUTPUT_FOLDER = "/media/hddb/mario/data/COMBINED" # ICAI
# OUTPUT_FOLDER = "/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets" # Picasso
OUTPUT_FOLDER = "/home/mariopasc/Python/Datasets/COMBINED"  # Sobremesa

# Root directories where the datasets are stored
# ROOT_DIR_SOURCE_DATASETS = "/home/mario/Python/Datasets/COMBINED/source" # Portátil

ROOT_DIR_SOURCE_DATASETS = (
    "/home/mariopasc/Python/Datasets/COMBINED/source"  # Sobremesa
)

# ARCADE task

ARCADE_TASK = "stenosis"

# Preprocessing steps to be performed on the datasets
PLAN_STEPS = {
    "resolution_standarization": {
        "desired_X": 512,
        "desired_Y": 512,
        "method": "bilinear",
    },
    "dtype_standarization": {"desired_dtype": "uint8"},
    "format_standarization": {"desired_format": "png"},
    # "clahe": {
    #    "window_size": 5,
    #    "sigma": 1.0,
    #    "clipLimit": 3.0,
    #    "tileGridSize": (8, 8),
    # },
    # "filtering_smoothing_equalization": {"window_size": 5, "sigma": 1.0},
    "dataset_formats": {
        "YOLO": True,
        "FasterRCNN": True,
        "RetinaNet": True,
        "SSD": True,
    },
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
    "CADICA": ROOT_DIR_SOURCE_DATASETS,
    "ARCADE": ROOT_DIR_SOURCE_DATASETS,
    "KEMEROVO": ROOT_DIR_SOURCE_DATASETS,
}

print("Integrating datasets...")
final_json: Dict[str, Any] = DatasetGenerator.integrate_datasets(
    DATASETS_TO_PROCESS, root_dirs, arcade_task=ARCADE_TASK
)
with open(output_combined_json, "w") as f:
    json.dump(final_json, f, indent=4)
print(f"Combined standardized JSON saved to {output_combined_json}")

# --- Preprocessing Planning Step ---
with open(output_combined_json, "r") as f:
    data = json.load(f)
print("Creating preprocessing plan...")
planned_data = DatasetGenerator.create_preprocessing_plan(data, PLAN_STEPS)
with open(output_planned_json, "w") as f:
    json.dump(planned_data, f, indent=4)
print(f"Preprocessing plan saved to {output_planned_json}")

# --- Preprocessing Execution Step ---
# Here, the user can supply a list of steps in order.
steps_order = list(PLAN_STEPS.keys())

output_ica_detection = os.path.join(OUTPUT_FOLDER, "ICA_DETECTION")
print("Applying preprocessing plan...")
DatasetGenerator.apply_preprocessing_plan(
    output_planned_json, output_ica_detection, steps_order
)
print("Preprocessing completed.")

# Move JSONs (Cleanup stage)

json_folder = os.path.join(output_ica_detection, "json_metadata")
os.makedirs(json_folder, exist_ok=True)

json_files = [
    os.path.join(OUTPUT_FOLDER, file)
    for file in os.listdir(OUTPUT_FOLDER)
    if file.endswith(".json")
]
json_files.append(os.path.join(output_ica_detection, "processed.json"))

for file in json_files:
    shutil.move(src=file, dst=os.path.join(json_folder, os.path.basename(file)))

print("Applying holdout to non-PyTorch datasets")
print(
    "[INFO] PyTorch datasets will be divided just like the other datasets\nbut they return DataLoader objects, so they must be splitted during trainig."
)

DatasetGenerator.execute_holdout_pipeline(
    root_folder=output_ica_detection,
    splits_dict=SPLITS_DICT,
    output_splits_json=os.path.join(json_folder, "splits.json"),
    include_datasets=DATASETS_TO_PROCESS,
)
