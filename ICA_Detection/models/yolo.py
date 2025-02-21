# ica_yolo_detection/models/yolo.py

import time
import logging
import os
from ICA_Detection.external.ultralytics.ultralytics import YOLO  # or adjust this import based on your project structure
from typing import Dict, Any

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("./yolo_training.log"), logging.StreamHandler()],
)
logging.info("Starting cross-validation process.")


class Detection_YOLO:
    """
    A class to manage the training, tuning, and validation processes of YOLO for object detection.

    Attributes:
        train_params (Dict[str, Any]): Dictionary of training hyperparameters.
        model (YOLO): YOLO model instance.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the Detection_YOLO class with the specified configuration.

        The configuration dictionary contains parameters for training such as:

            {
              "task": "detect",
              "mode": "train",
              "model": "/path/to/yolov8l.pt",
              "data": "./config.yaml",
              "epochs": 1000,
              "batch": 16,
              "imgsz": 640,
              ... (other hyperparameters)
            }

        Keys "model", "data", "device", "name", and "save_dir" are used for initialization,
        while all other keys are passed as hyperparameters to the training method.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        try:
            self.args = config
            self.model_path = self.args.get("model")
            self.data_yaml = self.args.get("data")
            self.device = self.args.get("device", "cuda:0")
            self.name = self.args.get("name", "ateroesclerosis_training")
            self.save_dir = self.args.get("save_dir", None)
            # Remove keys that are not meant for training.
            excluded_keys = {"model", "data", "device", "name", "save_dir", "task", "mode", "time"}
            self.train_params = {k: v for k, v in self.args.items() if k not in excluded_keys}
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
            # Pass additional parameters if necessary.
            self.model.val(
                data=self.data_yaml,
                imgsz=self.args.get("imgsz", 640),
                batch=self.args.get("batch", 16),
                iou=self.args.get("iou", 0.7),
                conf=0.001,
                plots=True,
                split=split,
                name=self.args.get("name", "ICA")
            )
            logging.info("Validation completed successfully on split: %s", split)
        except Exception as e:
            logging.error("Validation error: %s", e)
            raise
        total_time = time.time() - start_time
        logging.info("Total validation time on split '%s': %.2f seconds", split, total_time)


if __name__ == "__main__":
    # Hard-coded configuration parameters:
    config = {
        "task": "detect",
        "mode": "train",
        "model": "/home/mariopascual/Projects/CADICA/ICA_DETECTION/yolov8l.pt",
        "data": "/media/hddb/mario/data/COMBINED/yolo_ica_detection.yaml",
        "epochs": 100,
        "batch": 8,
        "imgsz": 512,
        "save": True,
        "save_period": 1,
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
        "name": "ICA_detection",
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
        "auto_augment": '',
        "erasing": 0.0,
        "crop_fraction": 0.0,
        "save_dir": "/home/mariopascual/Projects/CADICA/ICA_DETECTION/base_dataset_run"
    }
    split = "val"

    detector = Detection_YOLO(config)
    detector.train()
    detector.val(split=split)
