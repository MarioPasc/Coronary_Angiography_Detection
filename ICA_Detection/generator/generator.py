# ica_yolo_detection/generator.py

import json
from typing import Any, Dict, List
import os

# Import the integration function from the integration module.
from ICA_Detection.integration.integrate import integrate_datasets
# Import the preprocessing planner function.
from ICA_Detection.preprocessing.planner import create_preprocessing_plan

class DatasetGenerator:
    """
    DatasetGenerator serves as the main entry point for integrating datasets and planning preprocessing.
    
    It wraps the integration and preprocessing planning functions from the respective modules.
    """
    
    @staticmethod
    def integrate_datasets(datasets: List[str], root_dirs: Dict[str, str], arcade_task: str = "stenosis") -> Dict[str, Any]:
        """
        Integrate standardized JSON outputs from multiple datasets.
        
        Args:
            datasets (List[str]): List of dataset names to process, e.g., ["CADICA", "ARCADE", "KEMEROVO"].
            root_dirs (Dict[str, str]): Mapping from dataset name to its root directory.
            arcade_task (str, optional): For ARCADE, which task to process ("stenosis", "syntax", or "both"). Defaults to "stenosis".
        
        Returns:
            Dict[str, Any]: Combined standardized JSON with a top-level "Standard_dataset" key.
        """
        return integrate_datasets(datasets, root_dirs, arcade_task=arcade_task)
    
    @staticmethod
    def create_preprocessing_plan(data: Dict[str, Any], plan_steps: Dict[str, Any]) -> Dict[str, Any]:
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


if __name__ == "__main__":
    # --- Integration Step ---
    # You can download the datasets manually from:
    # - KEMEROV: https://data.mendeley.com/datasets/ydrm75xywg/2
    # - ARCADE: https://zenodo.org/records/10390295 
    # - CADICA: https://data.mendeley.com/datasets/p9bpx9ctcv/2 
    
    # Define the list of datasets to integrate.
    datasets_to_process = ["CADICA", "ARCADE", "KEMEROVO"]
    
    # Define standard folder to output results
    output_base_folder = "/home/mario/Python/Datasets/COMBINED"
    output_combined_json = os.path.join(output_base_folder, "combined_standardized.json")
    output_planned_json = os.path.join(output_base_folder, "planned_standardized.json")

    # Provide a mapping from dataset names to the required root directories.
    # Note: For CADICA and ARCADE, use the root folders expected by their processing functions.
    root_dirs = {
        "CADICA": "/home/mario/Python/Datasets",
        "ARCADE": "/home/mario/Python/Datasets",
        "KEMEROVO": "/home/mario/Python/Datasets"
    }
    
    # Optionally, specify the ARCADE task ("stenosis", "syntax", or "both").
    arcade_task = "stenosis"
    
    print("Integrating datasets...")
    final_json: Dict[str, Any] = DatasetGenerator.integrate_datasets(datasets_to_process, root_dirs, arcade_task=arcade_task)
    
    # Save the combined JSON to a file.
    with open(output_combined_json, "w") as f:
        json.dump(final_json, f, indent=4)
    print(f"Combined standardized JSON saved to {output_combined_json}")
    
    # --- Preprocessing Planning Step ---
    
    with open(output_combined_json, "r") as f:
        data = json.load(f)
    
    # Define all preprocessing steps in a single plan.
    plan_steps = {
        "format_standarization": {"desired_format": "png"},
        "dtype_standarization": {"desired_dtype": "uint8"},
        "resolution_standarization": {"desired_X": 512, "desired_Y": 512, "method": "bilinear"},
        #"filtering_smoothing_equalization": {"window_size": 5, "sigma": 1.0}
    }
    
    print("Creating preprocessing plan...")
    planned_data = DatasetGenerator.create_preprocessing_plan(data, plan_steps)
    
    with open(output_planned_json, "w") as f:
        json.dump(planned_data, f, indent=4)
    print(f"Preprocessing plan saved to {output_planned_json}")
