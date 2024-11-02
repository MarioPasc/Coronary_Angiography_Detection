import ultralytics as ultralytics
import os
import glob
import pandas as pd
import logging
from typing import Dict

# Configure logging
logging.basicConfig(filename='yolov8_detection.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Detection_YOLO:
    """
    A class to manage the training, tuning, and validation processes of YOLOv8 for object detection.

    Attributes
        yaml_path (str): Path to the data YAML file.
        model (ultralytics.YOLO): YOLO model instance.
    """

    def __init__(self, yaml_path: str, model_path: str = "yolov8l.pt") -> None:
        """
        Initializes the Detection_YOLOv8 class with the specified model and data configuration.

        Args:
            yaml_path (str): Path to the data YAML file.
            model_path (str): Path to the YOLO model file (default is 'yolov8l.pt').
        """
        try:
            self.model = ultralytics.YOLO(model=model_path)
            self.yaml_path = yaml_path
            logging.info("Initialized Detection_YOLOv8 with model path: %s and YAML path: %s", model_path, yaml_path)
        except Exception as e:
            logging.error("Error initializing Detection_YOLOv8: %s", e)
            raise

    def train(self, hyperparameters: Dict[str, float] = {}) -> None:
        """
        Trains the YOLO model with the specified hyperparameters.

        Args:
            hyperparameters (dict): Additional hyperparameters to override the default training parameters.

        Raises:
            Exception: If training fails due to model configuration or other issues.
        """
        try:
            # Default parameters
            default_params = {
                'data': self.yaml_path,
                'epochs': 100,
                'batch': 16,
                'imgsz': 640,
                'save': True,
                'save_period': 1,
                'project': None,
                'name': "ateroesclerosis_training",
                'pretrained': True,
                'optimizer': 'Adam',
                'iou': 0.7,
                'lr0': 7.5e-05,
                'lrf': 0.004,
                'momentum': 0.71,
                'weight_decay': 0.00024,
                'warmup_epochs': 4.5,
                'warmup_momentum': 0.83,
                'warmup_bias_lr': 0.025,
                'box': 9.38,
                'cls': 0.68361,
                'dfl': 1.69,
                'single_cls': True,
                'cos_lr': True,
                'augment': False,
                # Additional augmentations set to 0
                'degrees': 0.0,
                'translate': 0.0,
                'scale': 0.0,
                'shear': 0.0,
                'mosaic': 0.0,
                'copy_paste': 0.0,
            }
            params = {**default_params, **hyperparameters}
            logging.info("Starting training with parameters: %s", params)
            self.model.train(**params)
            logging.info("Training completed successfully.")
        except Exception as e:
            logging.error("Training error: %s", e)
            raise

    def tune(self) -> None:
        """
        Performs hyperparameter tuning for the YOLO model using default parameters.

        Raises:
            Exception: If tuning fails due to model configuration or other issues.
        """
        try:
            logging.info("Starting hyperparameter tuning.")
            results_tuning = self.model.tune(
                data=self.yaml_path, epochs=100, iterations=100, save=True, plots=True, val=True,
                name="ateroesclerosis_tuning", seed=42, single_cls=True, cos_lr=True,
                box=7.5, cls=0.5, dfl=1.5, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
                warmup_epochs=3, warmup_momentum=0.8, imgsz=640, optimizer='Adam', augment=False,
                crop_fraction=0.0, iou=0.5, degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
                perspective=0.0, flipud=0.0, fliplr=0.0, mosaic=0.0, close_mosaic=0, mixup=0.0,
                copy_paste=0.0, erasing=0.0
            )
            logging.info("Tuning completed successfully.")
        except Exception as e:
            logging.error("Tuning error: %s", e)
            raise

    def val(self) -> None:
        """
        Validates the model using the weights from each epoch and collects performance metrics.

        Raises:
            Exception: If validation fails due to model configuration or other issues.
        """
        try:
            weights_folder = "./runs/detect/ateroesclerosis_training/weights"
            results_list = []

            # Gather weight files, including 'last.pt' and 'best.pt'
            weight_files = sorted([file for file in glob.glob(os.path.join(weights_folder, '*.pt')) if 'last.pt' not in file])
            weight_files.append(os.path.join(weights_folder, 'last.pt'))
            weight_files.insert(0, os.path.join(weights_folder, 'best.pt'))

            for epoch, weight in enumerate(weight_files):
                epoch_num = 0 if os.path.basename(weight) == 'best.pt' else 100 if os.path.basename(weight) == 'last.pt' else int(os.path.basename(weight).replace('epoch', '').replace('.pt', ''))
                
                # Load and validate the model with the current weights
                model_batch = ultralytics.YOLO(weight)
                results = model_batch.val(imgsz=640, conf=0.01, plots=True)

                # Collect results
                precision_b = results.box.mp
                recall_b = results.box.mr
                map_05_b = results.box.map50
                map_05_95_b = results.box.map

                results_list.append({
                    "epoch": epoch_num,
                    "weight": os.path.basename(weight),
                    "precision": precision_b,
                    "recall": recall_b,
                    "map_50": map_05_b,
                    "map_50_95": map_05_95_b
                })
                logging.info("Validation completed for weight: %s", os.path.basename(weight))

            # Save validation results to a CSV file
            results_df = pd.DataFrame(results_list)
            results_csv_path = os.path.join('./runs/detect/ateroesclerosis_training', 'validation_results.csv')
            results_df.to_csv(results_csv_path, index=False)
            logging.info("Validation results saved to %s", results_csv_path)
            print("Validation results saved to 'validation_results.csv'")
        except Exception as e:
            logging.error("Validation error: %s", e)
            raise


def main() -> int:
    """
    Main function to create a Detection_YOLOv8 instance and perform training and validation.

    Returns:
        int: Exit code, 0 for success.
    """
    try:
        model = Detection_YOLO(model_path="", yaml_path="./config.yaml")
        model.train()
        model.val()
        # Uncomment to run hyperparameter tuning with Simulated annealing (https://en.wikipedia.org/wiki/Simulated_annealing). 
        # Ultralytics doesnt really implement a GA for this tuning.
        # model.tune()
        return 0
    except Exception as e:
        logging.error("Execution error in main: %s", e)
        return 1

if __name__ == "__main__":
    main()
