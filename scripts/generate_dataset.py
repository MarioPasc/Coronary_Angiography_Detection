import os
import json

from typing import Dict, Any

from ICA_Detection.generator.generator import DatasetGenerator

import logging
import shutil

# ==========================
# = USER-DEFINED VARIABLES =
# ==========================

# Define log file path
LOG_FILE = "./dataset_generation.log"  # Change this to your desired log file path

# Datasets to integrate and combined. Possible args incluide ["CADICA", "ARCADE", "KEMEROVO"]
DATASETS_TO_PROCESS = ["CADICA"]

# Splits dictionary. Split type and % of the images being allocated in that split
SPLITS_DICT = {"train": 0.67, "val": 0.33, "test": 0.0}
SEED = 42

# Output folder to store the final combined, preprocessed dataset and splits
# OUTPUT_FOLDER = "/media/hddb/mario/data/COMBINED" # ICAI
# OUTPUT_FOLDER = "/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets" # Picasso
OUTPUT_FOLDER = "/home/mpascual/research/datasets/angio/tasks"  # Sobremesa

# Root directories where the datasets are stored
ROOT_DIR_SOURCE_DATASETS = "/home/mpascual/research/datasets/angio/source"  # Portátil

# Preprocessing steps to be performed on the datasets

PLAN_STEPS_DETECTION = {
    "resolution_standarization": {
        "desired_X": 512,
        "desired_Y": 512,
        "method": "bilinear",
    },
    "dtype_standarization": {"desired_dtype": "uint8"},
    "format_standarization": {"desired_format": "png"},
    "dataset_formats": {"YOLO": True, "COCO": True},
}

PLAN_STEPS_SEGMENTATION = {
    "resolution_standarization": {
        "desired_X": 512,
        "desired_Y": 512,
        "method": "bilinear",
    },
    "clahe": {
        "window_size": 5,
        "sigma": 1.0,
        "clipLimit": 3.0,
        "tileGridSize": (8, 8),
    },
    "filtering_smoothing_equalization": {"window_size": 5, "sigma": 1.0},
    "dtype_standarization": {"desired_dtype": "uint8"},
    "format_standarization": {"desired_format": "png"},
    "dataset_formats": {"YOLO": True, "COCO": True},
}

"""
    "clahe": {
        "window_size": 5,
        "sigma": 1.0,
        "clipLimit": 3.0,
        "tileGridSize": (8, 8),
    },
    "filtering_smoothing_equalization": {"window_size": 5, "sigma": 1.0},
"""

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

detection_folder = os.path.join(OUTPUT_FOLDER, "stenosis_detection")
segmentation_folder = os.path.join(OUTPUT_FOLDER, "arteries_segmentation")
os.makedirs(detection_folder, exist_ok=True)
os.makedirs(segmentation_folder, exist_ok=True)

detection_folder_jsons = os.path.join(detection_folder, "json")
segmentation_folder_jsons = os.path.join(segmentation_folder, "json")
os.makedirs(detection_folder_jsons, exist_ok=True)
os.makedirs(segmentation_folder_jsons, exist_ok=True)

output_combined_detection = os.path.join(
    detection_folder_jsons, "combined_standardized.json"
)
output_combined_segmentation = os.path.join(
    segmentation_folder_jsons, "combined_standardized.json"
)

output_planned_detection = os.path.join(
    detection_folder_jsons, "planned_standardized.json"
)
output_planned_segmentation = os.path.join(
    segmentation_folder_jsons, "planned_standardized.json"
)

root_dirs = {
    "CADICA": ROOT_DIR_SOURCE_DATASETS,
    "ARCADE": ROOT_DIR_SOURCE_DATASETS,
    "KEMEROVO": ROOT_DIR_SOURCE_DATASETS,
}

if os.path.exists(output_combined_detection) and os.path.exists(output_combined_segmentation):
    print(f"Loading existing integrated datasets from {detection_folder_jsons}...")
    logger.info(f"Loading existing integrated datasets from {detection_folder_jsons}")
    with open(output_combined_detection, "r") as f:
        detection_json = json.load(f)
    with open(output_combined_segmentation, "r") as f:
        segmentation_json = json.load(f)
    print("Loaded existing integrated datasets.")
    logger.info("Loaded existing integrated datasets.")
else:
    print("Integrating datasets...")
    logger.info("Integrating datasets...")
    final_json: Dict[str, Any] = DatasetGenerator.integrate_datasets(
        DATASETS_TO_PROCESS, root_dirs
    )

    # We must separate the detection task from the segmentation task.
    detection_json: Dict[str, Any] = final_json.get("detection", [])
    segmentation_json: Dict[str, Any] = final_json.get("segmentation", [])

    with open(output_combined_detection, "w") as f:
        json.dump(detection_json, f, indent=4)
    print(f"Detection JSON saved to {output_combined_detection}")
    logger.info(f"Detection JSON saved to {output_combined_detection}")
    with open(output_combined_segmentation, "w") as f:
        json.dump(segmentation_json, f, indent=4)
    print(f"Segmentation JSON saved to {output_combined_segmentation}")
    logger.info(f"Segmentation JSON saved to {output_combined_segmentation}")

# --- Preprocessing Planning Step ---
if os.path.exists(output_planned_detection) and os.path.exists(output_planned_segmentation):
    print(f"Loading existing preprocessing plans from {detection_folder_jsons}...")
    logger.info(f"Loading existing preprocessing plans from {detection_folder_jsons}")
    with open(output_planned_detection, "r") as f:
        planned_data_detection = json.load(f)
    with open(output_planned_segmentation, "r") as f:
        planned_data_segmentation = json.load(f)
    print("Loaded existing preprocessing plans.")
    logger.info("Loaded existing preprocessing plans.")
else:
    # Ensure data_detection and data_segmentation are loaded if not already from the integration step
    if 'detection_json' not in locals(): # Check if loaded from existing combined files
        with open(output_combined_detection, "r") as f:
            data_detection = json.load(f)
    else:
        data_detection = detection_json

    if 'segmentation_json' not in locals(): # Check if loaded from existing combined files
        with open(output_combined_segmentation, "r") as f:
            data_segmentation = json.load(f)
    else:
        data_segmentation = segmentation_json

    print("Creating preprocessing plan for detection...")
    logger.info("Creating preprocessing plan for detection...")
    planned_data_detection = DatasetGenerator.create_preprocessing_plan(
        data_detection, PLAN_STEPS_DETECTION, root_name="Stenosis_Detection"
    )
    print("Creating preprocessing plan for segmentation...")
    logger.info("Creating preprocessing plan for segmentation...")
    planned_data_segmentation = DatasetGenerator.create_preprocessing_plan(
        data_segmentation, PLAN_STEPS_SEGMENTATION, root_name="Arteries_Segmentation"
    )

    with open(output_planned_detection, "w") as f:
        json.dump(planned_data_detection, f, indent=4)
    print(f"Preprocessing plan saved to {output_planned_detection}")
    logger.info(f"Preprocessing plan saved to {output_planned_detection}")
    with open(output_planned_segmentation, "w") as f:
        json.dump(planned_data_segmentation, f, indent=4)
    print(f"Preprocessing plan saved to {output_planned_segmentation}")
    logger.info(f"Preprocessing plan saved to {output_planned_segmentation}")


# --- Preprocessing Execution Step ---
steps_order_detection = list(PLAN_STEPS_DETECTION.keys())
steps_order_segmentation = list(PLAN_STEPS_SEGMENTATION.keys())

# Check for existence of key output directories as a proxy for completion
detection_images_dir = os.path.join(detection_folder, "images")
detection_labels_dir = os.path.join(detection_folder, "labels") # Assuming labels are also generated here

if os.path.exists(detection_images_dir) and os.path.isdir(detection_images_dir) and os.listdir(detection_images_dir) and \
   os.path.exists(detection_labels_dir) and os.path.isdir(detection_labels_dir): # Add more checks if needed, e.g., os.listdir not empty
    print("Skipping preprocessing execution for detection as output directories seem to exist.")
    logger.info("Skipping preprocessing execution for detection as output directories seem to exist.")
else:
    print("Applying preprocessing plan for detection...")
    logger.info("Applying preprocessing plan for detection...")
    DatasetGenerator.apply_preprocessing_plan(
        output_planned_detection, detection_folder, steps_order_detection
    )
    print("Preprocessing for detection completed.")
    logger.info("Preprocessing for detection completed.")

segmentation_images_dir = os.path.join(segmentation_folder, "images")
segmentation_labels_dir = os.path.join(segmentation_folder, "labels") # Assuming labels are also generated here

if os.path.exists(segmentation_images_dir) and os.path.isdir(segmentation_images_dir) and os.listdir(segmentation_images_dir) and \
   os.path.exists(segmentation_labels_dir) and os.path.isdir(segmentation_labels_dir): # Add more checks if needed
    print("Skipping preprocessing execution for segmentation as output directories seem to exist.")
    logger.info("Skipping preprocessing execution for segmentation as output directories seem to exist.")
else:
    print("Applying preprocessing plan for segmentation...")
    logger.info("Applying preprocessing plan for segmentation...")
    DatasetGenerator.apply_preprocessing_plan(
        output_planned_segmentation, segmentation_folder, steps_order_segmentation
    )
    print("Preprocessing for segmentation completed.")
    logger.info("Preprocessing for segmentation completed.")

path_arteries_arcade = os.path.join(ROOT_DIR_SOURCE_DATASETS, "ARCADE", "images")
if os.path.exists(path_arteries_arcade):
    # This rmtree might be problematic if ARCADE is part of DATASETS_TO_PROCESS
    # and its source is needed again. Consider if this cleanup is always desired.
    # For now, keeping it as is, but it's a point of attention.
    print(f"Cleaning up {path_arteries_arcade}...")
    logger.info(f"Cleaning up {path_arteries_arcade}...")
    shutil.rmtree(path=path_arteries_arcade)

print("Applying holdout to non-PyTorch datasets")
logger.info("Applying holdout to non-PyTorch datasets")
print(
    "[INFO] PyTorch datasets will be divided just like the other datasets\nbut they return DataLoader objects, so they must be splitted during trainig."
)
logger.info("[INFO] PyTorch datasets will be divided just like the other datasets but they return DataLoader objects, so they must be splitted during trainig.")

output_splits_detection_json = os.path.join(detection_folder, "json", "splits.json")
if os.path.exists(output_splits_detection_json):
    print(f"Skipping holdout pipeline for detection as {output_splits_detection_json} already exists.")
    logger.info(f"Skipping holdout pipeline for detection as {output_splits_detection_json} already exists.")
else:
    DatasetGenerator.execute_holdout_pipeline(
        root_folder=detection_folder,
        splits_dict=SPLITS_DICT,
        output_splits_json=output_splits_detection_json,
        include_datasets=DATASETS_TO_PROCESS,
        seed=SEED,
    )
    logger.info(f"Holdout pipeline for detection completed. Splits saved to {output_splits_detection_json}")

output_splits_segmentation_json = os.path.join(segmentation_folder, "json", "splits.json")
if os.path.exists(output_splits_segmentation_json):
    print(f"Skipping holdout pipeline for segmentation as {output_splits_segmentation_json} already exists.")
    logger.info(f"Skipping holdout pipeline for segmentation as {output_splits_segmentation_json} already exists.")
else:
    DatasetGenerator.execute_holdout_pipeline(
        root_folder=segmentation_folder,
        splits_dict=SPLITS_DICT,
        output_splits_json=output_splits_segmentation_json,
        include_datasets=DATASETS_TO_PROCESS,
        seed=SEED,
    )
    logger.info(f"Holdout pipeline for segmentation completed. Splits saved to {output_splits_segmentation_json}")