#!/usr/bin/env python3

# This script evaluates the YOLO model's performance on different labels by creating
# label-specific datasets and running validations on them.

import os
import logging
import yaml
import pandas as pd
from typing import Union, List, Dict
from ultralytics import YOLO
import torch.multiprocessing

# Set up logging
os.makedirs('./logs', exist_ok=True)
logging.basicConfig(filename='./logs/test_by_labels.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting test by labels.")

def load_config(yaml_path: str) -> dict:
    """
    Loads configuration parameters from a YAML file into a dictionary.
    """
    try:
        with open(yaml_path, 'r') as file:
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
    csv_dir = os.path.join(config['OUTPUT_PATH'], 'CADICA_Holdout_Info')  # Path to the CSV files
    splits = ['train', 'val', 'test']
    data_frames = {}

    for split in splits:
        csv_path = os.path.join(csv_dir, f'{split}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            data_frames[split] = df
            logging.info(f"Loaded {split} CSV with {len(df)} entries.")
        else:
            logging.error(f"CSV file for {split} split not found at {csv_path}")
            raise FileNotFoundError(f"CSV file for {split} split not found at {csv_path}")

    return data_frames

def create_label_datasets(data_frames: Dict[str, pd.DataFrame], output_base_dir: str, path_to_YOLO_dataset: str) -> List[str]:
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
        labels.update(df['LesionLabel'].unique())

    labels = list(labels)
    logging.info(f"Unique labels found: {labels}")

    for label in labels:
        label_dir = os.path.join(output_base_dir, f'CADICA_{label}')
        images_dir = os.path.join(label_dir, 'images')
        labels_dir = os.path.join(label_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for split in ['train', 'val', 'test']:
            split_images_dir = os.path.join(label_dir, 'images', split)
            split_labels_dir = os.path.join(label_dir, 'labels', split)
            os.makedirs(split_images_dir, exist_ok=True)
            os.makedirs(split_labels_dir, exist_ok=True)

            df_split = data_frames.get(split, pd.DataFrame())
            df_label = df_split[df_split['LesionLabel'] == label]
            logging.info(f"Processing {len(df_label)} images for label '{label}' in split '{split}'.")

            for idx, row in df_label.iterrows():
                frame_path = row['Frame_path']
                filename = os.path.basename(frame_path)  # e.g., 'p1_v2_00015.png'

                # Construct paths to image and label in the YOLO dataset
                image_src = os.path.join(path_to_YOLO_dataset, 'images', split, filename)
                label_filename = os.path.splitext(filename)[0] + '.txt'
                label_src = os.path.join(path_to_YOLO_dataset, 'labels', split, label_filename)

                # Destination paths
                image_dst = os.path.join(split_images_dir, filename)
                label_dst = os.path.join(split_labels_dir, label_filename)

                # Create symlink or copy the image
                if os.path.exists(image_src):
                    if not os.path.exists(image_dst):
                        try:
                            os.symlink(image_src, image_dst)
                        except Exception as e:
                            logging.error(f"Failed to create symlink for image {image_src} to {image_dst}: {e}")

                if label.lower() == 'nolesion':
                    # For 'nolesion', create an empty label file if it doesn't exist
                    if not os.path.exists(label_dst):
                        try:
                            with open(label_dst, 'w') as f:
                                pass  # Create an empty file
                            logging.info(f"Created empty label file for 'nolesion' at {label_dst}")
                        except Exception as e:
                            logging.error(f"Failed to create empty label file at {label_dst}: {e}")
                else:
                    # Copy or symlink the label file
                    if os.path.exists(label_src):
                        if not os.path.exists(label_dst):
                            try:
                                os.symlink(label_src, label_dst)
                            except Exception as e:
                                logging.error(f"Failed to create symlink for label {label_src} to {label_dst}: {e}")

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
            'path': os.path.join(output_base_dir, f'CADICA_{label}'),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {0: label}
        }
        config_path = os.path.join(output_base_dir, f'config_{label}.yaml')
        with open(config_path, 'w') as file:
            yaml.dump(config, file)
        logging.info(f"Created config file for label '{label}' at {config_path}")

def run_validation_on_labels(labels: List[str], output_base_dir: str, model_path: str) -> pd.DataFrame:
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
    splits = ['train', 'val', 'test']
    for label in labels:
        logging.info(f"Validating model on label '{label}' dataset.")
        config_path = os.path.join(output_base_dir, f'config_{label}.yaml')
        model = YOLO(model=model_path, task="detect", verbose=True)

        label_results = {'Label': label}
        for split in splits:
            logging.info(f"Validating on split '{split}' for label '{label}'.")
            # Run validation
            val = model.val(data=config_path, imgsz=640, batch=8, iou=0.6, plots=False, split=split, workers=0)
            # Collect results
            label_results[f'{split}/mAP50'] = val.box.map50
            label_results[f'{split}/mAP50-95'] = val.box.map
        results.append(label_results)

    df_results = pd.DataFrame(results)
    return df_results

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    CONFIG_PATH = "./config.yaml"
    CONFIG = load_config(CONFIG_PATH)

    # Define paths
    output_base_dir = CONFIG['OUTPUT_PATH']  
    models = ['../models/simulated_annealing.pt',
              '../models/iteration1.pt',
              '../models/iteration2.pt']  

    # Define path to YOLO dataset
    path_to_YOLO_dataset = os.path.join(CONFIG['OUTPUT_PATH'], CONFIG["YOLO_DATASET_FOLDER_NAME"])

    if not os.path.exists(path_to_YOLO_dataset):
        logging.error(f"Path to YOLO Dataset not found at {path_to_YOLO_dataset}")
        raise FileNotFoundError(f"Path to YOLO Dataset not found at {path_to_YOLO_dataset}")

    # Load CSV files
    data_frames = load_csv_files(CONFIG)

    # Create label-specific datasets
    labels = create_label_datasets(data_frames, output_base_dir, path_to_YOLO_dataset)

    # Create config files for each label
    create_config_files(labels, output_base_dir)

    for model in models:
        # Run validation on each label-specific dataset
        df_results = run_validation_on_labels(labels, output_base_dir, model)

        # Save results to CSV
        model_name = os.path.basename(model).strip('.pt')
        results_csv_path = os.path.join(output_base_dir, f'test_label_model_{model_name}.csv')
        df_results.to_csv(results_csv_path, index=False)
        logging.info(f"Validation results saved to {results_csv_path}")
