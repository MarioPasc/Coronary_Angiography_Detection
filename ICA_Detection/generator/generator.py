# ICA_Detection/generator/generator.py

import json
import os
from typing import Any, Dict, List
import shutil

# Import the integration function from the integration module.
from ICA_Detection.integration.integrate import integrate_datasets

# Import the preprocessing planner function.
from ICA_Detection.preprocessing.planner import create_preprocessing_plan

# Import the process_images function from the preprocessing module.
from ICA_Detection.preprocessing.preprocessing import process_images_by_task

# Import Holdout functions

from ICA_Detection.splits.holdout.holdout_global import holdout
from ICA_Detection.splits.holdout.holdout_yolo import apply_holdout_yolo


class DatasetGenerator:
    """
    DatasetGenerator serves as the main entry point for integrating datasets, planning preprocessing,
    and applying the preprocessing steps.

    It wraps the integration, preprocessing planning, and execution functions from the respective modules.
    """

    @staticmethod
    def integrate_datasets(
        datasets: List[str], root_dirs: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Integrate standardized JSON outputs from multiple datasets.

        Args:
            datasets (List[str]): List of dataset names to process, e.g., ["CADICA", "ARCADE", "KEMEROVO"].
            root_dirs (Dict[str, str]): Mapping from dataset name to its root directory.
            arcade_task (str, optional): For ARCADE, which task to process ("stenosis", "syntax", or "both").
                                         Defaults to "stenosis".

        Returns:
            Dict[str, Any]: Combined standardized JSON with a top-level "Standard_dataset" key.
        """
        return integrate_datasets(datasets, root_dirs)

    @staticmethod
    def create_preprocessing_plan(
        data: Dict[str, Any], plan_steps: Dict[str, Any], root_name: str
    ) -> Dict[str, Any]:
        """
        Create a preprocessing plan for the dataset based on specified plan steps.

        The plan_steps dictionary should include keys for the preprocessing steps to be applied. For example:

            {
              "resolution_standarization": {"desired_X": 512, "desired_Y": 512, "method": "bilinear"},
              "dtype_standarization": {"desired_dtype": "uint8"},
              "format_standarization": {"desired_format": "png"},
              "filtering_smoothing_equalization": {"window_size": 5, "sigma": 1.0}
            }

        Args:
            data (Dict[str, Any]): Standardized JSON dataset.
            plan_steps (Dict[str, Any]): Dictionary of preprocessing steps and parameters.

        Returns:
            Dict[str, Any]: The updated JSON dataset with a "preprocessing_plan" field for each entry.
        """
        return create_preprocessing_plan(data, plan_steps, root_name)

    @staticmethod
    def apply_preprocessing_plan(
        planned_json_path: str, output_folder: str, steps_order: List[str]
    ) -> None:
        """
        Apply the preprocessing plan to all images in the dataset.

        This method reads the planned standardized JSON file (which contains the "preprocessing_plan" field)
        and creates an output folder structure:
            output_folder/
                images/
                labels/
        It then applies the preprocessing steps in the specified order:
          1. format_standarization: Convert image to PNG if necessary.
          2. dtype_standarization: Standardize the image data type.
          3. resolution_standarization: Resize the image and re-normalize bounding boxes.
          4. filtering_smoothing_equalization: Apply filtering-smoothing equalization.

        For images with "lesion": true, the annotations are saved in YOLO format
        (one bounding box per row: "stenosis x_center y_center width height") in the labels folder.

        Args:
            planned_json_path (str): Path to the planned standardized JSON file.
            output_folder (str): Base output folder where the structure "ICA_DETECTION/images" and "ICA_DETECTION/labels" will be created.
            steps_order (List[str]): List of preprocessing steps in the desired order.
        """
        process_images_by_task(planned_json_path, output_folder, steps_order)

    @staticmethod
    def execute_holdout_pipeline(
        root_folder: str,
        splits_dict: Dict[str, float],
        output_splits_json: str,
        include_datasets: List[str],
        seed: int = 42
    ):
        """
        High-level pipeline to:
        1) Perform a one-time holdout on images_folder => produce splits_info.json.
        2) Apply that split to create YOLO subfolders (train/val/test).
        3) Apply that same split to create PyTorch subfolders for Faster R-CNN, RetinaNet, etc.

        :param root_folder: High-level folder for reference (if needed).
        :param splits_dict: e.g. {"train": 0.7, "val": 0.3, "test": 0.0}
        :param output_splits_json: e.g. "splits_info.json"
        :param include_datasets: Filter list, e.g. ["CADICA"]
        """

        # 1) Generate splits_info.json
        #    (This does NOT create symlinks or subfolders. It just partitions patients.)
        print("Performing holdout to generate splits_info.json...")
        # holdout(...) is a function that groups by patient,
        # randomly assigns them to train/val/test, and saves a JSON structure.

        images_folder = os.path.join(root_folder, "images")

        holdout(
            images_folder=images_folder,
            splits=splits_dict,
            output_json_path=output_splits_json,
            include_datasets=include_datasets,
            seed = seed
        )
        # 2) Apply YOLO holdout
        #    This reads the splits_info.json and creates subfolders: images/train, images/val, ...
        # apply_holdout_yolo(splits_info_json, input_root, output_root)
        dataset_folder = os.path.join(root_folder, "datasets")
        if "yolo" in os.listdir(path=dataset_folder):
            print("Applying YOLO holdout splits...")
            apply_holdout_yolo(
                splits_info_path=output_splits_json,
                input_root=os.path.join(
                    dataset_folder, "yolo"
                ),  # contains images/ and labels/ in YOLO format
                output_root=os.path.join(
                    dataset_folder, "yolo"
                ),  # e.g. "datasets/yolo" or similar
                yaml_filename="yolo_ica_detection.yaml",
            )