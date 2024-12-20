#!/usr/bin/env python3

# This script evaluates the YOLO model's performance on different labels by creating
# label-specific datasets and running validations on them.

import os
import logging
import yaml
import pandas as pd
from typing import Union, List, Dict
from CADICA_Detection.external.ultralytics.ultralytics import YOLO
import torch.multiprocessing

# Set up logging
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    filename="./logs/test_by_labels.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Starting test by labels.")


def load_config(yaml_path: str) -> dict:
    """
    Loads configuration parameters from a YAML file into a dictionary.
    """
    try:
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Error loading config.yaml file: {e}")
        raise e


def load_csv_files(config: dict) -> Dict[str, pd.DataFrame]:
    """
    Loads CSV files for train, val, and test splits.

    Returns:
        dict: A dictionary containing DataFrames for each split.
    """
    csv_dir = os.path.join(
        config["OUTPUT_PATH"], "CADICA_Holdout_Info"
    )  # Path to the CSV files
    splits = ["train", "val", "test"]
    data_frames = {}

    for split in splits:
        csv_path = os.path.join(csv_dir, f"{split}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            data_frames[split] = df
            logging.info(f"Loaded {split} CSV with {len(df)} entries.")
        else:
            logging.error(f"CSV file for {split} split not found at {csv_path}")
            raise FileNotFoundError(
                f"CSV file for {split} split not found at {csv_path}"
            )

    return data_frames


def create_label_datasets(
    data_frames: Dict[str, pd.DataFrame],
    output_base_dir: str,
    path_to_YOLO_dataset: str,
) -> List[str]:
    """
    Creates label-specific datasets in YOLO format.

    Args:
        data_frames (dict): DataFrames for each data split.
        output_base_dir (str): Base directory to store label-specific datasets.
        path_to_YOLO_dataset (str): Path to the main YOLO dataset.

    Returns:
        List[str]: List of labels for which datasets were created.
    """
    labels = set()
    for df in data_frames.values():
        for label_list in df["LesionLabel"]:
            # Split multi-labels and add them to the label set
            split_labels = label_list.split(",")  # Assuming comma-separated
            labels.update(split_labels)

    labels = list(labels)
    logging.info(f"Unique labels found: {labels}")

    for label in labels:
        label_dir = os.path.join(output_base_dir, f"CADICA_{label}")
        images_dir = os.path.join(label_dir, "images")
        labels_dir = os.path.join(label_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for split in ["train", "val", "test"]:
            split_images_dir = os.path.join(label_dir, "images", split)
            split_labels_dir = os.path.join(label_dir, "labels", split)
            os.makedirs(split_images_dir, exist_ok=True)
            os.makedirs(split_labels_dir, exist_ok=True)

            df_split = data_frames.get(split, pd.DataFrame())

            # Iterate through rows and duplicate image per corresponding label
            for _, row in df_split.iterrows():
                frame_path = row["Frame_path"]
                filename = os.path.basename(frame_path)

                split_labels = row["LesionLabel"].split(",")
                if label in split_labels:
                    # Image Source and Destination Paths
                    image_src = os.path.join(
                        path_to_YOLO_dataset, "images", split, filename
                    )
                    label_filename = os.path.splitext(filename)[0] + ".txt"
                    label_src = os.path.join(
                        path_to_YOLO_dataset, "labels", split, label_filename
                    )

                    image_dst = os.path.join(split_images_dir, filename)
                    label_dst = os.path.join(split_labels_dir, label_filename)

                    # Symlink or Copy Images and Labels
                    if os.path.exists(image_src) and not os.path.exists(image_dst):
                        os.symlink(image_src, image_dst)
                    if os.path.exists(label_src) and not os.path.exists(label_dst):
                        os.symlink(label_src, label_dst)

    return labels


def create_config_files(labels: List[str], output_base_dir: str) -> None:
    """
    Creates config.yaml files for each label-specific dataset.

    Args:
        labels (List[str]): List of labels.
        output_base_dir (str): Base directory where datasets are stored.
    """
    for label in labels:
        config = {
            "path": os.path.join(output_base_dir, f"CADICA_{label}"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: label},
        }
        config_path = os.path.join(output_base_dir, f"config_{label}.yaml")
        with open(config_path, "w") as file:
            yaml.dump(config, file)
        logging.info(f"Created config file for label '{label}' at {config_path}")


def run_validation_on_labels(
    labels: List[str], output_base_dir: str, model_path: str
) -> pd.DataFrame:
    """
    Runs validation on each label-specific dataset and collects results for each split (train, val, test).

    Args:
        labels (List[str]): List of labels.
        output_base_dir (str): Base directory where datasets are stored.
        model_path (str): Path to the trained YOLO model.

    Returns:
        pd.DataFrame: DataFrame containing validation results for each label and split.
    """
    results = []
    splits = ["train", "val", "test"]
    for label in labels:
        logging.info(f"Validating model on label '{label}' dataset.")
        config_path = os.path.join(output_base_dir, f"config_{label}.yaml")
        model = YOLO(model=model_path, task="detect", verbose=True)

        label_results = {"Label": label}
        for split in splits:
            logging.info(f"Validating on split '{split}' for label '{label}'.")
            # Run validation
            val = model.val(
                data=config_path,
                imgsz=512,
                batch=16,
                iou=0.5,
                plots=False,
                split=split,
                workers=0,
                half=True,
                device="cuda:1"
            )
            torch.cuda.empty_cache()
            # Collect results
            label_results[f"{split}/precision"] = val.box.mp
            label_results[f"{split}/recall"] = val.box.mr
            label_results[f"{split}/mAP50"] = val.box.map50
            label_results[f"{split}/mAP50-95"] = val.box.map
        results.append(label_results)

    df_results = pd.DataFrame(results)
    return df_results


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    CONFIG_PATH = "./scripts/config_sobremesa.yaml"
    CONFIG = load_config(CONFIG_PATH)

    # Define paths
    output_base_dir = "/home/mariopasc/Python/Results/Coronariografias/patient-based"
    models = [
        os.path.join(
            output_base_dir, "TPE", "detect", "trial_121_training", "weights", "best.pt"
        ),
        os.path.join(
            output_base_dir,
            "CMAES",
            "detect",
            "trial_131_training",
            "weights",
            "best.pt",
        ),
        os.path.join(
            output_base_dir,
            "RandomSamplerBaseline",
            "detect",
            "trial_22_training",
            "weights",
            "best.pt",
        ),
        os.path.join(output_base_dir, "Baseline", "weights", "best.pt"),
    ]

    model_names = ["TPE", "CMAES", "RANDOM", "BASELINE"]

    # Define path to YOLO dataset
    path_to_YOLO_dataset = os.path.join(
        CONFIG["OUTPUT_PATH"], CONFIG["YOLO_DATASET_FOLDER_NAME"]
    )

    if not os.path.exists(path_to_YOLO_dataset):
        logging.error(f"Path to YOLO Dataset not found at {path_to_YOLO_dataset}")
        raise FileNotFoundError(
            f"Path to YOLO Dataset not found at {path_to_YOLO_dataset}"
        )

    # Load CSV files
    data_frames = load_csv_files(CONFIG)

    # Create label-specific datasets
    labels = create_label_datasets(data_frames, output_base_dir, path_to_YOLO_dataset)

    # Create config files for each label
    create_config_files(labels, output_base_dir)

    idx = 0
    for model in models:
        # Run validation on each label-specific dataset
        df_results = run_validation_on_labels(labels, output_base_dir, model)

        # Save results to CSV
        model_name = model_names[idx]
        results_csv_path = os.path.join(
            output_base_dir, f"test_label_model_{model_name}.csv"
        )
        df_results.to_csv(results_csv_path, index=False)
        logging.info(f"Validation results saved to {results_csv_path}")
        idx += 1
