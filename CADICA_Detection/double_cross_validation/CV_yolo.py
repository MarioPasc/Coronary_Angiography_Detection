import os
import yaml
from typing import Dict, Any
from pathlib import Path

import pandas as pd
from ultralytics import YOLO
import torch
from labelTesterCV import LabelTester

import logging

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./cross_validation.log"),
        logging.StreamHandler()
    ]
)
logging.info("Starting cross-validation process.")

def load_config(config_path: str):
    """Load configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_hyperparameters(hyperparameters_path: str) -> Dict[str, Any]:
    """Load hyperparameters from YAML file."""
    with open(hyperparameters_path, "r") as f:
        return yaml.safe_load(f)

class Detection_YOLO:
    """
    A class to manage the YOLOv8 model for training and validation.
    """

    def __init__(self, yaml_path: str, model_path: str = "yolov8l.pt") -> None:
        """
        Initialize the YOLO model with a given configuration.

        Args:
            yaml_path (str): Path to the fold-specific config file.
            model_path (str): Path to the YOLO model.
        """
        self.model = YOLO(model=model_path)
        self.yaml_path = yaml_path
        self.config = load_config(yaml_path)
        self.dataset_path = self.config.get("path", "")

    def train(self, hyperparameters: Dict[str, Any], device: str) -> None:
        """
        Train the YOLO model with the provided hyperparameters.

        Args:
            hyperparameters (Dict[str, Any]): Hyperparameters for training.
            device (str): The device to use for training (e.g., "cuda:0").
        """
        try:
            params = {
                "data": self.yaml_path,
                "device": device,
                "name": hyperparameters.pop("name", "default_training"),
            }
            params.update(hyperparameters)
            self.model.train(**params)
            logging.info(f"Training completed for: {self.yaml_path}")
        except Exception as e:
            logging.error(f"Training error for {self.yaml_path}: {e}")

    def validate(self, unique_name: str, split: str = "test", device: str = "cuda:0") -> None:
        """
        Run validation using the best model on the specified data split.

        Args:
            unique_name (str): Unique name of the fold.
            split (str): Data split to validate (default is "test").
            device (str): The device to use for validation.
        """
        try:
            best_model_path = Path(f"./runs/detect/{unique_name}/weights/best.pt")
            if not best_model_path.exists():
                logging.error(f"Best model not found at {best_model_path}")
                return

            logging.info(f"Loading best model from {best_model_path}")
            model = YOLO(model=str(best_model_path), task="detect", verbose=True)

            logging.info(f"Running validation on split: {split}")
            val = model.val(
                data=self.config,
                imgsz=512,
                batch=16,
                iou=0.5,
                plots=False,
                split=split,
                workers=0,
                half=True,
                device=device,
            )

            torch.cuda.empty_cache()

            results = {
                f"{split}/precision": val.box.mp,
                f"{split}/recall": val.box.mr,
                f"{split}/mAP50": val.box.map50,
                f"{split}/mAP50-95": val.box.map,
            }

            results_csv_path = Path(f"./results/{unique_name}_validation_metrics.csv")
            pd.DataFrame([results]).to_csv(results_csv_path, index=False)
            logging.info(f"Validation metrics saved to {results_csv_path}")
        except Exception as e:
            logging.error(f"Validation error for split {split}: {e}")


def main():
    fold_configs_dir = "/media/hddb/mario/data/YOLO_Splits"
    splits_base_dir = "/media/hddb/mario/data/double_cv_splits"
    model_path = "yolov8l.pt"
    output_base_dir = "/media/hddb/mario/data/"
    device = "cuda:0"
    epochs = 100
    hyperparameters_path = "./args_tpe.yaml"  

    tuned_hyperparameters = load_hyperparameters(hyperparameters_path)


    logging.info(f"CUDA is available: {torch.cuda.is_available()}")
    logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    configs = [path for path in os.listdir(fold_configs_dir) if not os.path.isdir(os.path.join(fold_configs_dir, path))]

    logging.info(f"Config files found: ")
    for config in sorted(configs): logging.info(f"  - {config}")

    # Train and validate for each fold config
    for config_file in sorted(configs):
        config_path = Path(fold_configs_dir) / config_file
        unique_name = Path(config_file).stem

        logging.info(f"Processing fold: {unique_name}")
        detection_yolo = Detection_YOLO(yaml_path=str(config_path), model_path=model_path)

        # Determine corresponding CSV paths based on the config file name
        outer_fold = unique_name.split("_")[1]
        fold_dir = Path(splits_base_dir) / f"fold_{outer_fold}"
        internal_fold_dir = fold_dir / "internal_folds/internal_fold_1"

        train_csv_path = internal_fold_dir / "train.csv"
        val_csv_path = internal_fold_dir / "val.csv"
        test_csv_path = fold_dir / "test.csv"

        logging.info(f"Using train CSV: {train_csv_path}")
        logging.info(f"Using val CSV: {val_csv_path}")
        logging.info(f"Using test CSV: {test_csv_path}")

        # Override hyperparameters for the specific fold
        hyperparameters = tuned_hyperparameters.copy()
        hyperparameters["data"] = config_path
        hyperparameters["name"] = unique_name
        hyperparameters["epochs"] = epochs 

        # Train the model
        detection_yolo.train(hyperparameters, device=device)

        # Validate on the test split
        detection_yolo.validate(split="test", device=device)

        # Run label-specific validation
        if all(path.exists() for path in [train_csv_path, val_csv_path, test_csv_path]):
            train_df = pd.read_csv(train_csv_path)
            val_df = pd.read_csv(val_csv_path)
            test_df = pd.read_csv(test_csv_path)

            label_tester = LabelTester(train_df, val_df, test_df, detection_yolo.dataset_path, output_base_dir, model_path)
            labels = label_tester.create_label_datasets()
            label_tester.create_config_files(labels)
            results_df = label_tester.run_validation_on_labels(labels)

            results_csv_path = Path(output_base_dir) / f"{unique_name}_label_validation_results.csv"
            results_df.to_csv(results_csv_path, index=False)
            logging.info(f"Label-specific validation results saved to {results_csv_path}")
        else:
            logging.warning(f"Missing CSV files for fold: {unique_name}")

if __name__ == "__main__":
    main()
