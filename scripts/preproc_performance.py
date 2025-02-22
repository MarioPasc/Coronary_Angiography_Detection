"""
preproc_performance.py

This script automates the workflow of:
  1) Integrating (standardizing) CADICA, ARCADE, or both datasets.
  2) Applying two different preprocessing schemes:
       - RAW: Basic resizing, dtype, and format standardization.
       - CLAHE+FSE: Same as RAW plus CLAHE and filtering-smoothing equalization.
  3) Creating holdout splits for YOLO (train=0.7, val=0.3, test=0.0).
  4) Training a YOLOv8 model on each dataset combination + preprocessing scheme.
  5) Validating on the best checkpoint for that run.
  6) Logging Precision, Recall, F1, and mAP scores in a CSV file.
  7) Removing (cleaning up) all generated intermediate files/folders between each step
     to free disk space and ensure a fresh start for each experiment.

Usage:
  python preproc_performance.py

Dependencies / Project Structure:
  - This script expects the following modules and structure to be available:
      ICA_Detection/generator/generator.py         (DatasetGenerator)
      ICA_Detection/splits/holdout.py              (create_holdout_split)
      ica_yolo_detection/models/yolo.py            (Detection_YOLO)
      And the associated YOLOv8 libraries, etc.
"""

import os
import sys
import shutil
import json
import logging
import csv
import random
from pathlib import Path
from typing import List, Dict, Any

try:
    from ICA_Detection.generator.generator import DatasetGenerator
    from ICA_Detection.splits.holdout import create_holdout_split
    from ICA_Detection.models.yolo import Detection_YOLO
except ImportError as e:
    print(
        "Make sure to adjust the import statements to match your project's folder structure."
    )
    raise e

# --------------------------------------------------------------------------------
# Global Configuration
# --------------------------------------------------------------------------------

# Root directories for original (already downloaded) datasets
ROOT_DIRS = {
    "CADICA": "/media/hddb/mario/data/COMBINED",
    "ARCADE": "/media/hddb/mario/data/COMBINED",
}

# The base folder where integrated/preprocessed data will be temporarily saved.
# The script removes these intermediate folders after each run to free space.
OUTPUT_BASE_FOLDER = "/media/hddb/mario/data/COMBINED"

# ARCADE default task
ARCADE_TASK = "stenosis"

# Splits dictionary (train=0.7, val=0.3, test=0.0)
SPLITS_DICT = {"train": 0.7, "val": 0.3, "test": 0.0}

# YOLO hyperparameters (applied to all runs)
YOLO_CONFIG = {
    "task": "detect",
    "mode": "train",
    "model": "/home/mariopascual/Projects/CADICA/ICA_DETECTION/Execution/yolov8l.pt",  # Adjust if needed
    "epochs": 100,
    "batch": 8,
    "imgsz": 512,
    "save": True,
    "save_period": -1,
    "pretrained": True,
    "optimizer": "RAdam",
    "iou": 0.5,
    "lr0": 1.0e-05,
    "lrf": 0.005,
    "momentum": 0.7642866153393313,
    "weight_decay": 1.0e-05,
    "warmup_epochs": 5,
    "warmup_momentum": 0.8786382652173892,
    "warmup_bias_lr": 0.1,
    "box": 8.5,
    "cls": 0.65,
    "dfl": 0.2,
    "single_cls": True,
    "cos_lr": True,
    "augment": False,
    "device": "cuda:0",
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "bgr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",
    "auto_augment": "",
    "erasing": 0.0,
    "crop_fraction": 0.0,
}

# Preprocessing plans
PLAN_STEPS_RAW = {
    "resolution_standarization": {
        "desired_X": 512,
        "desired_Y": 512,
        "method": "bilinear",
    },
    "dtype_standarization": {"desired_dtype": "uint8"},
    "format_standarization": {"desired_format": "png"},
    "labels_formats": {"YOLO": True},
}

PLAN_STEPS_CLAHE_FSE = {
    "resolution_standarization": {
        "desired_X": 512,
        "desired_Y": 512,
        "method": "bilinear",
    },
    "dtype_standarization": {"desired_dtype": "uint8"},
    "format_standarization": {"desired_format": "png"},
    "clahe": {"window_size": 5, "sigma": 1.0, "clipLimit": 3.0, "tileGridSize": (8, 8)},
    "filtering_smoothing_equalization": {"window_size": 5, "sigma": 1.0},
    "labels_formats": {"YOLO": True},
}

# Random seed for reproducible splits
SEED = 42

# --------------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------


def set_random_seed(seed: int) -> None:
    """
    Sets the random seed for the Python built-in random module.
    (If you want, also set np.random.seed if you do other random steps with NumPy.)
    """
    random.seed(seed)
    # If needed, also do: np.random.seed(seed)
    logger.info(f"Random seed set to {seed}.")


def remove_dir_if_exists(path: str) -> None:
    """
    Safely remove a directory if it exists.
    """
    dir_path = Path(path)
    if dir_path.exists() and dir_path.is_dir():
        logger.info(f"Removing directory: {path}")
        shutil.rmtree(dir_path)


def remove_file_if_exists(path: str) -> None:
    """
    Safely remove a file if it exists.
    """
    file_path = Path(path)
    if file_path.exists() and file_path.is_file():
        logger.info(f"Removing file: {path}")
        file_path.unlink()


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


def integrate_and_preprocess(
    datasets: List[str],
    output_folder: str,
    plan_steps: Dict[str, Any],
    arcade_task: str = "stenosis",
) -> None:
    """
    Integrate the specified datasets into a single JSON, plan the preprocessing,
    and apply it to create the folder structure:
      output_folder/ICA_DETECTION/images
      output_folder/ICA_DETECTION/labels
    (along with any transformations defined in the plan_steps).
    Also renames/cleans up the YOLO labels folder as requested.
    """
    combined_json_path = os.path.join(output_folder, "combined_standardized.json")
    planned_json_path = os.path.join(output_folder, "planned_standardized.json")
    ica_detection_out = os.path.join(output_folder, "ICA_DETECTION")

    # 1) Integration
    logger.info(f"Integrating datasets: {datasets} into {combined_json_path}")
    final_json = DatasetGenerator.integrate_datasets(
        datasets, ROOT_DIRS, arcade_task=arcade_task
    )
    with open(combined_json_path, "w") as f:
        json.dump(final_json, f, indent=4)

    # 2) Preprocessing planning
    logger.info(
        f"Creating preprocessing plan for {datasets}, plan steps: {list(plan_steps.keys())}"
    )
    with open(combined_json_path, "r") as f:
        data = json.load(f)

    planned_data = DatasetGenerator.create_preprocessing_plan(data, plan_steps)
    with open(planned_json_path, "w") as f:
        json.dump(planned_data, f, indent=4)

    # 3) Preprocessing execution
    logger.info(f"Applying preprocessing plan for {datasets}.")
    steps_order = list(plan_steps.keys())
    DatasetGenerator.apply_preprocessing_plan(
        planned_json_path, ica_detection_out, steps_order
    )

    # Clean up the labels folder naming
    rename_labels_folder(ica_detection_out)


def create_splits_and_train(
    datasets: List[str],
    output_folder: str,
    experiment_name: str,
    results_collector: List[Dict[str, Any]],
) -> None:
    """
    1) Create holdout split (train=0.7, val=0.3, test=0.0) in YOLO_ICA_DETECTION
    2) Train YOLO
    3) Validate on best checkpoint
    4) Log metrics to results_collector
    """
    ica_detection_in = os.path.join(output_folder, "ICA_DETECTION")
    yolo_detection_out = os.path.join(output_folder, "YOLO_ICA_DETECTION")
    yaml_filename = "dataset.yaml"

    # 1) Create holdout split
    logger.info(f"Creating holdout split for {datasets} at {yolo_detection_out}")
    remove_dir_if_exists(yolo_detection_out)  # Ensure a fresh start

    create_holdout_split(
        input_root=ica_detection_in,
        splits=SPLITS_DICT,
        output_root=yolo_detection_out,
        include_datasets=datasets,
        yaml_filename=yaml_filename,
        splits_info_filename="splits_info.json",
    )

    # 2) Train YOLO
    data_yaml_path = os.path.join(yolo_detection_out, yaml_filename)
    run_dir = "runs/detect"  # YOLO default root for detection runs
    YOLO_CONFIG["data"] = data_yaml_path
    YOLO_CONFIG["name"] = experiment_name
    YOLO_CONFIG["save_dir"] = run_dir

    logger.info(f"Starting YOLO training for experiment: {experiment_name}")
    detector = Detection_YOLO(YOLO_CONFIG)
    detector.train()  # train with the given hyperparams

    # 3) Validate on the best checkpoint
    best_ckpt = os.path.join(run_dir, experiment_name, "weights", "best.pt")
    os.rename(
        src=os.path.join(run_dir, "train"), dst=os.path.join(run_dir, experiment_name)
    )
    logger.info(f"Validating with best checkpoint: {best_ckpt}")

    # Overwrite the model path for validation
    detector.model = None  # Clean reference
    detector.model_path = best_ckpt

    # Re-initialize YOLO to use best weights
    # We'll re-use the same config, adjusting mode and model:
    yolo_val_config = YOLO_CONFIG.copy()
    yolo_val_config["mode"] = "val"
    yolo_val_config["model"] = best_ckpt
    yolo_val_config["data"] = data_yaml_path
    yolo_val_config["name"] = experiment_name
    yolo_val_config["save_dir"] = run_dir

    val_detector = Detection_YOLO(yolo_val_config)
    # the .val() method in the snippet doesn't return anything, so let's modify it here:
    # We'll call the underlying ultralytics YOLO directly to capture metrics:
    metrics = val_detector.val(split="val")

    shutil.move(
        src=os.path.join(run_dir, experiment_name + "2"),
        dst=os.path.join(run_dir, experiment_name),
    )

    # 4) Collect metrics

    metrics_dict = {}
    # The actual attribute names can differ by version. Adjust as needed.
    metrics_dict["precision"] = metrics.box.mp
    metrics_dict["recall"] = metrics.box.mr

    p = metrics_dict["precision"]
    r = metrics_dict["recall"]
    metrics_dict["f1"] = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

    metrics_dict["map50"] = metrics.box.map50
    metrics_dict["map50_95"] = metrics.box.map

    logger.info(f"Validation Metrics for {experiment_name}: {metrics_dict}")
    results_collector.append(
        {
            "experiment_name": experiment_name,
            "datasets": "+".join(datasets),
            "precision": metrics_dict["precision"],
            "recall": metrics_dict["recall"],
            "f1": metrics_dict["f1"],
            "map50": metrics_dict["map50"],
            "map50_95": metrics_dict["map50_95"],
        }
    )


def save_results_to_csv(results: List[Dict[str, Any]], csv_path: str) -> None:
    """
    Saves the results dictionary list to a CSV file.
    If the file does not exist, it creates one; otherwise, it overwrites it.
    """
    if not results:
        logger.info("No results to save. The results list is empty.")
        return

    fieldnames = list(results[0].keys())
    logger.info(f"Saving results to CSV: {csv_path}")

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def run_experiment_scenario(
    scenario_name: str,
    plan_steps: Dict[str, Any],
    dataset_combinations: List[List[str]],
    csv_collector: List[Dict[str, Any]],
) -> None:
    """
    Orchestrates the workflow for a given scenario (RAW or CLAHE+FSE):
      - For each dataset combination:
         1) Integrate and preprocess
         2) Create splits, train, validate
         3) Collect metrics
         4) Clean up intermediate data
    """
    logger.info(f"=== Starting scenario: {scenario_name} ===")

    output_folder = os.path.join(OUTPUT_BASE_FOLDER, f"{scenario_name}_WORKDIR")
    remove_dir_if_exists(output_folder)  # Fresh start

    # We'll integrate & preprocess data for *all* chosen datasets at once,
    # then create splits for each subset. But user wants to do them one by one,
    # so let's do them in a loop:
    for combo in dataset_combinations:
        # 1) Integrate & Preprocess
        logger.info(f"[{scenario_name}] Generating dataset for: {combo}")
        os.makedirs(output_folder, exist_ok=True)
        integrate_and_preprocess(
            datasets=combo, output_folder=output_folder, plan_steps=plan_steps
        )

        # 2) Create splits, train, validate
        # experiment_name can be something like "CADICA_RAW", "ARCADE_CLAHEFSE", etc.
        combo_name = "".join(combo)  # e.g. ["CADICA", "ARCADE"] -> "CADICAARCADE"
        experiment_name = f"{combo_name}_{scenario_name}"
        create_splits_and_train(
            datasets=combo,
            output_folder=output_folder,
            experiment_name=experiment_name,
            results_collector=csv_collector,
        )

        # 3) Remove the YOLO_ICA_DETECTION splits folder to free space, but keep runs/detect logs
        yolo_detection_out = os.path.join(output_folder, "YOLO_ICA_DETECTION")
        remove_dir_if_exists(yolo_detection_out)

        # 4) Remove the ICA_DETECTION folder as well, plus combined_standardized.json, planned_standardized.json
        ica_detection_out = os.path.join(output_folder, "ICA_DETECTION")
        remove_dir_if_exists(ica_detection_out)
        remove_file_if_exists(os.path.join(output_folder, "combined_standardized.json"))
        remove_file_if_exists(os.path.join(output_folder, "planned_standardized.json"))

    # Finally, remove the entire scenario output folder
    # to free up space for next scenario
    remove_dir_if_exists(output_folder)


def main() -> None:
    """
    Main entry point: runs two scenarios (RAW, CLAHE+FSE) across 3 dataset combos:
      1) [CADICA]
      2) [ARCADE]
      3) [CADICA, ARCADE]

    Gathers results, writes them to a CSV file, and cleans up intermediate folders.
    """
    # 1) Configure seed
    set_random_seed(SEED)

    # 2) Setup final results collector
    results: List[Dict[str, Any]] = []

    # 3) Define dataset combos
    #    (We exclude KEMEROVO for now; only CADICA, ARCADE, CADICA+ARCADE)
    combos = [["CADICA"], ["ARCADE"], ["CADICA", "ARCADE"]]

    # 4) Run RAW scenario
    run_experiment_scenario("RAW", PLAN_STEPS_RAW, combos, results)

    # 5) Run CLAHE+FSE scenario
    run_experiment_scenario("CLAHEFSE", PLAN_STEPS_CLAHE_FSE, combos, results)

    # 6) Save final results to CSV
    results_csv_path = os.path.join(
        OUTPUT_BASE_FOLDER, "preprocessing_performance_results.csv"
    )
    save_results_to_csv(results, results_csv_path)
    logger.info("All experiments completed.")


if __name__ == "__main__":
    main()
