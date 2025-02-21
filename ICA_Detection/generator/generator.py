# ica_yolo_detection/generator.py

import json
import os
from typing import Any, Dict, List

# Import the integration function from the integration module.
from ICA_Detection.integration.integrate import integrate_datasets

# Import the preprocessing planner function.
from ICA_Detection.preprocessing.planner import create_preprocessing_plan

# Import the process_images function from the preprocessing module.
from ICA_Detection.preprocessing.preprocessing import process_images


class DatasetGenerator:
    """
    DatasetGenerator serves as the main entry point for integrating datasets, planning preprocessing,
    and applying the preprocessing steps.

    It wraps the integration, preprocessing planning, and execution functions from the respective modules.
    """

    @staticmethod
    def integrate_datasets(
        datasets: List[str], root_dirs: Dict[str, str], arcade_task: str = "stenosis"
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
        return integrate_datasets(datasets, root_dirs, arcade_task=arcade_task)

    @staticmethod
    def create_preprocessing_plan(
        data: Dict[str, Any], plan_steps: Dict[str, Any]
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
        return create_preprocessing_plan(data, plan_steps)

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
        process_images(planned_json_path, output_folder, steps_order)


if __name__ == "__main__":
    # --- Integration Step ---
    # You can download the datasets manually from:
    # - KEMEROV: https://data.mendeley.com/datasets/ydrm75xywg/2
    # - ARCADE: https://zenodo.org/records/10390295
    # - CADICA: https://data.mendeley.com/datasets/p9bpx9ctcv/2

    # datasets_to_process = ["CADICA", "ARCADE", "KEMEROVO"]
    datasets_to_process = ["CADICA"]

    output_base_folder = "/media/hddb/mario/data/COMBINED"
    os.makedirs(output_base_folder, exist_ok=True)
    output_combined_json = os.path.join(
        output_base_folder, "combined_standardized.json"
    )
    output_planned_json = os.path.join(output_base_folder, "planned_standardized.json")

    root_dirs_local = {
        "CADICA": "/home/mario/Python/Datasets",
        "ARCADE": "/home/mario/Python/Datasets",
        "KEMEROVO": "/home/mario/Python/Datasets",
    }

    root_dirs = {
        "CADICA": "/media/hddb/mario/data/COMBINED",
        "ARCADE": "/media/hddb/mario/data/COMBINED",
        "KEMEROVO": "/media/hddb/mario/data/COMBINED",
    }

    

    arcade_task = "stenosis"

    print("Integrating datasets...")
    final_json: Dict[str, Any] = DatasetGenerator.integrate_datasets(
        datasets_to_process, root_dirs, arcade_task=arcade_task
    )
    with open(output_combined_json, "w") as f:
        json.dump(final_json, f, indent=4)
    print(f"Combined standardized JSON saved to {output_combined_json}")

    # --- Preprocessing Planning Step ---
    with open(output_combined_json, "r") as f:
        data = json.load(f)
    plan_steps = {
        "resolution_standarization": {
            "desired_X": 512,
            "desired_Y": 512,
            "method": "bilinear",
        },
        "dtype_standarization": {"desired_dtype": "uint8"},
        "format_standarization": {"desired_format": "png"},
        "clahe":  {"window_size": 5, "sigma": 1.0, "clipLimit": 3.0, "tileGridSize": (8,8)},
        "filtering_smoothing_equalization": {"window_size": 5, "sigma": 1.0},
        "labels_formats": {"YOLO": True},  # New key for additional label generation.
    }
    print("Creating preprocessing plan...")
    planned_data = DatasetGenerator.create_preprocessing_plan(data, plan_steps)
    with open(output_planned_json, "w") as f:
        json.dump(planned_data, f, indent=4)
    print(f"Preprocessing plan saved to {output_planned_json}")

    # --- Preprocessing Execution Step ---
    # Here, the user can supply a list of steps in order.
    steps_order = list(plan_steps.keys())

    output_ica_detection = os.path.join(output_base_folder, "ICA_DETECTION")
    print("Applying preprocessing plan...")
    DatasetGenerator.apply_preprocessing_plan(
        output_planned_json, output_ica_detection, steps_order
    )
    print("Preprocessing completed.")
