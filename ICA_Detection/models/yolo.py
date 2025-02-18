# ica_yolo_detection/models/yolo.py

import time
import logging
import yaml
from ICA_Detection.external.ultralytics.ultralytics import (
    YOLO,
)

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("./yolo_training.log"), logging.StreamHandler()],
)


class Detection_YOLO:
    """
    A class to manage training, tuning, and validation of YOLO for object detection.

    Attributes:
        yaml_path (str): Path to the data YAML file (used during training).
        model (YOLO): YOLO model instance.
        train_params (Dict[str, Any]): Dictionary of training hyperparameters.
    """

    def __init__(self, args_yaml: str) -> None:
        """
        Initialize the YOLO detector from a YAML configuration file.

        The YAML file contains parameters for training such as:

        ```yaml
        task: detect
        mode: train
        model: /path/to/yolov8l.pt
        data: ./config.yaml
        epochs: 1000
        batch: 16
        imgsz: 512
        ... (other hyperparameters)
        ```

        Keys "model", "data", "device", "name", and "save_dir" are used for initialization,
        while all other keys are passed as hyperparameters to the training method.

        Args:
            args_yaml (str): Path to the YAML configuration file.
        """
        try:
            with open(args_yaml, "r") as f:
                self.args = yaml.safe_load(f)
            self.model_path = self.args.get("model")
            self.data_yaml = self.args.get("data")
            self.device = self.args.get("device")
            self.name = self.args.get("name")
            self.save_dir = self.args.get("save_dir")
            # Remove keys that are not meant for training.
            excluded_keys = {
                "model",
                "data",
                "device",
                "name",
                "save_dir",
                "task",
                "mode",
                "time",
            }
            self.train_params = {
                k: v for k, v in self.args.items() if k not in excluded_keys
            }
            self.train_params["data"] = self.data_yaml  # Ensure 'data' is included.
            # Initialize YOLO model.
            self.model = YOLO(model=self.model_path)
            logging.info("Initialized YOLO model with model path: %s", self.model_path)
        except Exception as e:
            logging.error("Error initializing Detection_YOLO: %s", e)
            raise

    def train(self) -> None:
        """
        Train the YOLO model using the hyperparameters from the configuration.
        Measures and logs the total training time.
        """
        start_time = time.time()
        logging.info("Starting training with parameters: %s", self.train_params)
        try:
            self.model.train(**self.train_params)
            logging.info("Training completed successfully.")
        except Exception as e:
            logging.error("Training error: %s", e)
            raise
        total_time = time.time() - start_time
        logging.info("Total training time: %.2f seconds", total_time)

    def val(self, split: str) -> None:
        """
        Validate the YOLO model on the specified data split.
        Measures and logs the total validation time.

        Args:
            split (str): The split to use for validation (e.g., "train", "val", or "test").
        """
        start_time = time.time()
        logging.info("Starting validation on split: %s", split)
        try:
            # Assuming self.model.val accepts a 'split' parameter.
            self.model.val(
                data=self.data_yaml,
                imgsz=self.args.get("imgsz"),
                batch=self.args.get("batch"),
                iou=self.args.get("iou"),
                conf=0.01,
                plots=True,
                split=split,
            )
            logging.info("Validation completed successfully on split: %s", split)
        except Exception as e:
            logging.error("Validation error: %s", e)
            raise
        total_time = time.time() - start_time
        logging.info(
            "Total validation time on split '%s': %.2f seconds", split, total_time
        )


if __name__ == "__main__":
    logging.info("Starting YOLO training and validation process.")
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and validate YOLO model using YAML configuration."
    )
    parser.add_argument(
        "--args_yaml",
        required=True,
        help="Path to YAML file with training configuration.",
    )
    parser.add_argument(
        "--val_split",
        default="val",
        help="Data split to use for validation (e.g., 'train', 'val', 'test').",
    )
    args = parser.parse_args()

    detector = Detection_YOLO(args_yaml=args.args_yaml)
    detector.train()
    detector.val(split=args.val_split)
