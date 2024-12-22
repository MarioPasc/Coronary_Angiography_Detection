import os
import yaml
from CADICA_Detection.external.ultralytics.ultralytics import YOLO
from pathlib import Path
from typing import Dict

def load_config(config_path: str):
    """Load configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class Detection_YOLO:
    """
    A class to manage the YOLOv8 model for training.
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

    def train(self, hyperparameters: Dict[str, float], device: str) -> None:
        """
        Train the YOLO model with the provided hyperparameters.

        Args:
            hyperparameters (Dict[str, float]): Hyperparameters for training.
            device (str): The device to use for training (e.g., "cuda:0").
        """
        try:
            # Override default training parameters
            params = {
                "data": self.yaml_path,
                "device": device,  # User-specified device
                "name": hyperparameters.pop("name", "default_training"),
            }
            params.update(hyperparameters)  # Add remaining hyperparameters
            self.model.train(**params)
            print(f"Training completed for: {self.yaml_path}")
        except Exception as e:
            print(f"Training error for {self.yaml_path}: {e}")

def main():
    fold_configs_dir = "/home/mariopasc/Python/Datasets/CADICA_Project/YOLO_Splits"
    model_path = "yolov8l.pt"  # Fixed model path
    device = "cuda:0"

    # Load configurations
    ARGS = load_config("args.yaml")

    # Hyperparameters from args.yaml
    hyperparameters = {
        k: v for k, v in ARGS.items() if k not in ["model", "device", "data"]
    }

    # Train the model for each fold config
    for config_file in sorted(os.listdir(fold_configs_dir)):
        config_path = Path(fold_configs_dir) / config_file

        # Parse fold name from the config filename
        fold_name_parts = Path(config_file).stem.split("_")
        if len(fold_name_parts) == 4:  # Expected format: outer_X_inner_X
            outer_fold = fold_name_parts[1]
            inner_fold = fold_name_parts[3]
            unique_name = f"outer_{outer_fold}_inner_{inner_fold}"
        else:
            unique_name = Path(config_file).stem  # Fallback in case of unexpected naming

        # Initialize Detection_YOLO with the current config
        detection_yolo = Detection_YOLO(
            yaml_path=str(config_path), model_path=model_path
        )

        # Assign the unique name for this run
        hyperparameters["name"] = unique_name

        print(f"Training with config: {config_path} on device: {device} with name: {unique_name}")
        detection_yolo.train(hyperparameters, device=device)

if __name__ == "__main__":
    main()
