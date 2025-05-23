import os
import json
import yaml
import argparse
import logging
import shutil
from typing import Dict, Any, List, Tuple

from ICA_Detection.generator.generator import DatasetGenerator
from ICA_Detection.tools.prints_dsgen import print_welcome_and_structure
import time

# ==========================
# ========= SCRIPT =========
# ==========================

def setup_logging(log_file_path: str) -> logging.Logger:
    """Sets up the logger."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Prevent duplicate handlers if script is run multiple times in the same session
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_path, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Convert tileGridSize to tuple if present, as YAML loads it as a list
    if 'clahe' in config.get('preprocessing', {}).get('segmentation', {}).get('steps', {}):
        tile_grid_size = config['preprocessing']['segmentation']['steps']['clahe'].get('tileGridSize')
        if isinstance(tile_grid_size, list):
            config['preprocessing']['segmentation']['steps']['clahe']['tileGridSize'] = tuple(tile_grid_size)
    return config

def main(config: Dict[str, Any]):
    """Main function to run the dataset generation pipeline."""
    start_time = time.time() # <--- RECORD START TIME

    #print_welcome_and_structure()
    # Extract parameters from config
    log_file = config['logging']['log_file']
    datasets_to_process = config['dataset_processing']['datasets_to_process']
    splits_dict = config['splitting']['splits_dict']
    seed = config['splitting']['seed']
    output_folder = config['dataset_processing']['output_folder']
    root_dir_source_datasets = config['dataset_processing']['root_dir_source_datasets']
    
    plan_steps_detection = config['preprocessing']['detection']['steps']
    plan_name_detection = config['preprocessing']['detection']['plan_name']
    plan_steps_segmentation = config['preprocessing']['segmentation']['steps']
    plan_name_segmentation = config['preprocessing']['segmentation']['plan_name']

    logger = setup_logging(log_file)

    os.makedirs(output_folder, exist_ok=True)

    detection_folder = os.path.join(output_folder, "stenosis_detection")
    segmentation_folder = os.path.join(output_folder, "arteries_segmentation")
    os.makedirs(detection_folder, exist_ok=True)
    os.makedirs(segmentation_folder, exist_ok=True)

    detection_folder_jsons = os.path.join(detection_folder, "json")
    segmentation_folder_jsons = os.path.join(segmentation_folder, "json")
    os.makedirs(detection_folder_jsons, exist_ok=True)
    os.makedirs(segmentation_folder_jsons, exist_ok=True)

    output_combined_detection = os.path.join(
        detection_folder_jsons, "combined_standardized.json"
    )
    output_combined_segmentation = os.path.join(
        segmentation_folder_jsons, "combined_standardized.json"
    )

    output_planned_detection = os.path.join(
        detection_folder_jsons, "planned_standardized.json"
    )
    output_planned_segmentation = os.path.join(
        segmentation_folder_jsons, "planned_standardized.json"
    )

    root_dirs = {
        dataset: root_dir_source_datasets for dataset in ["CADICA", "ARCADE", "KEMEROVO"]
    } # Assuming all source datasets are in the same root_dir_source_datasets

    if os.path.exists(output_combined_detection) and os.path.exists(output_combined_segmentation):
        print(f"Loading existing integrated datasets from {detection_folder_jsons}...")
        logger.info(f"Loading existing integrated datasets from {detection_folder_jsons}")
        with open(output_combined_detection, "r") as f:
            detection_json = json.load(f)
        with open(output_combined_segmentation, "r") as f:
            segmentation_json = json.load(f)
        print("Loaded existing integrated datasets.")
        logger.info("Loaded existing integrated datasets.")
    else:
        print("Integrating datasets...")
        logger.info("Integrating datasets...")
        final_json: Dict[str, Any] = DatasetGenerator.integrate_datasets(
            datasets_to_process, root_dirs
        )

        detection_json: Dict[str, Any] = final_json.get("detection", {}) # Use .get with default
        segmentation_json: Dict[str, Any] = final_json.get("segmentation", {}) # Use .get with default

        with open(output_combined_detection, "w") as f:
            json.dump(detection_json, f, indent=4)
        print(f"Detection JSON saved to {output_combined_detection}")
        logger.info(f"Detection JSON saved to {output_combined_detection}")
        with open(output_combined_segmentation, "w") as f:
            json.dump(segmentation_json, f, indent=4)
        print(f"Segmentation JSON saved to {output_combined_segmentation}")
        logger.info(f"Segmentation JSON saved to {output_combined_segmentation}")

    # --- Preprocessing Planning Step ---
    if os.path.exists(output_planned_detection) and os.path.exists(output_planned_segmentation):
        print(f"Loading existing preprocessing plans from {detection_folder_jsons}...")
        logger.info(f"Loading existing preprocessing plans from {detection_folder_jsons}")
        with open(output_planned_detection, "r") as f:
            planned_data_detection = json.load(f)
        with open(output_planned_segmentation, "r") as f:
            planned_data_segmentation = json.load(f)
        print("Loaded existing preprocessing plans.")
        logger.info("Loaded existing preprocessing plans.")
    else:
        data_detection_for_planning = detection_json if 'detection_json' in locals() and detection_json else {}
        if not data_detection_for_planning and os.path.exists(output_combined_detection):
             with open(output_combined_detection, "r") as f:
                data_detection_for_planning = json.load(f)
        
        data_segmentation_for_planning = segmentation_json if 'segmentation_json' in locals() and segmentation_json else {}
        if not data_segmentation_for_planning and os.path.exists(output_combined_segmentation):
            with open(output_combined_segmentation, "r") as f:
                data_segmentation_for_planning = json.load(f)

        print("Creating preprocessing plan for detection...")
        logger.info("Creating preprocessing plan for detection...")
        planned_data_detection = DatasetGenerator.create_preprocessing_plan(
            data_detection_for_planning, plan_steps_detection, root_name=plan_name_detection
        )
        print("Creating preprocessing plan for segmentation...")
        logger.info("Creating preprocessing plan for segmentation...")
        planned_data_segmentation = DatasetGenerator.create_preprocessing_plan(
            data_segmentation_for_planning, plan_steps_segmentation, root_name=plan_name_segmentation
        )

        with open(output_planned_detection, "w") as f:
            json.dump(planned_data_detection, f, indent=4)
        print(f"Preprocessing plan saved to {output_planned_detection}")
        logger.info(f"Preprocessing plan saved to {output_planned_detection}")
        with open(output_planned_segmentation, "w") as f:
            json.dump(planned_data_segmentation, f, indent=4)
        print(f"Preprocessing plan saved to {output_planned_segmentation}")
        logger.info(f"Preprocessing plan saved to {output_planned_segmentation}")

    # --- Preprocessing Execution Step ---
    steps_order_detection = list(plan_steps_detection.keys())
    steps_order_segmentation = list(plan_steps_segmentation.keys())

    detection_images_dir = os.path.join(detection_folder, "images")
    detection_labels_dir = os.path.join(detection_folder, "labels")

    if os.path.exists(detection_images_dir) and os.path.isdir(detection_images_dir) and os.listdir(detection_images_dir) and \
       os.path.exists(detection_labels_dir) and os.path.isdir(detection_labels_dir) and os.listdir(detection_labels_dir):
        print("Skipping preprocessing execution for detection as output directories seem to exist and are not empty.")
        logger.info("Skipping preprocessing execution for detection as output directories seem to exist and are not empty.")
    else:
        print("Applying preprocessing plan for detection...")
        logger.info("Applying preprocessing plan for detection...")
        DatasetGenerator.apply_preprocessing_plan(
            output_planned_detection, detection_folder, steps_order_detection
        )
        print("Preprocessing for detection completed.")
        logger.info("Preprocessing for detection completed.")

    segmentation_images_dir = os.path.join(segmentation_folder, "images")
    segmentation_labels_dir = os.path.join(segmentation_folder, "labels")

    if os.path.exists(segmentation_images_dir) and os.path.isdir(segmentation_images_dir) and os.listdir(segmentation_images_dir) and \
       os.path.exists(segmentation_labels_dir) and os.path.isdir(segmentation_labels_dir) and os.listdir(segmentation_labels_dir):
        print("Skipping preprocessing execution for segmentation as output directories seem to exist and are not empty.")
        logger.info("Skipping preprocessing execution for segmentation as output directories seem to exist and are not empty.")
    else:
        print("Applying preprocessing plan for segmentation...")
        logger.info("Applying preprocessing plan for segmentation...")
        DatasetGenerator.apply_preprocessing_plan(
            output_planned_segmentation, segmentation_folder, steps_order_segmentation
        )
        print("Preprocessing for segmentation completed.")
        logger.info("Preprocessing for segmentation completed.")

    # Conditional cleanup for ARCADE, only if it's not in datasets_to_process or handled carefully
    # This part needs careful consideration based on how DatasetGenerator uses source files.
    # If ARCADE is processed and its source files are modified or moved by DatasetGenerator,
    # this cleanup might be fine. Otherwise, it could delete needed source data.
    # For now, let's assume it's a post-processing cleanup specific to ARCADE's structure.
    if "ARCADE" in datasets_to_process: # Or a more specific condition
        path_arteries_arcade = os.path.join(root_dir_source_datasets, "ARCADE", "images") # Assuming this is the path to clean
        if os.path.exists(path_arteries_arcade):
            print(f"Cleaning up {path_arteries_arcade}...")
            logger.info(f"Cleaning up {path_arteries_arcade}...")
            # shutil.rmtree(path=path_arteries_arcade) # Be cautious with this line.
            logger.warning(f"Cleanup of {path_arteries_arcade} was skipped. Uncomment if sure.")


    print("Applying holdout to non-PyTorch datasets")
    logger.info("Applying holdout to non-PyTorch datasets")
    print(
        "[INFO] PyTorch datasets will be divided just like the other datasets\n"
        "but they return DataLoader objects, so they must be splitted during training."
    )
    logger.info(
        "[INFO] PyTorch datasets will be divided just like the other datasets "
        "but they return DataLoader objects, so they must be splitted during training."
    )

    output_splits_detection_json = os.path.join(detection_folder_jsons, "splits.json")
    if os.path.exists(output_splits_detection_json):
        print(f"Skipping holdout pipeline for detection as {output_splits_detection_json} already exists.")
        logger.info(f"Skipping holdout pipeline for detection as {output_splits_detection_json} already exists.")
    else:
        DatasetGenerator.execute_holdout_pipeline(
            root_folder=detection_folder,
            splits_dict=splits_dict,
            output_splits_json=output_splits_detection_json,
            include_datasets=datasets_to_process, # This should ideally come from the processed data, not the initial list
            seed=seed,
        )
        logger.info(f"Holdout pipeline for detection completed. Splits saved to {output_splits_detection_json}")

    output_splits_segmentation_json = os.path.join(segmentation_folder_jsons, "splits.json")
    if os.path.exists(output_splits_segmentation_json):
        print(f"Skipping holdout pipeline for segmentation as {output_splits_segmentation_json} already exists.")
        logger.info(f"Skipping holdout pipeline for segmentation as {output_splits_segmentation_json} already exists.")
    else:
        DatasetGenerator.execute_holdout_pipeline(
            root_folder=segmentation_folder,
            splits_dict=splits_dict,
            output_splits_json=output_splits_segmentation_json,
            include_datasets=datasets_to_process, # Similar to above, review this parameter's source
            seed=seed,
        )
        logger.info(f"Holdout pipeline for segmentation completed. Splits saved to {output_splits_segmentation_json}")


    end_time = time.time() # <--- RECORD END TIME
    execution_time = end_time - start_time
    print(f"\n===================================================================================")
    print(f"Dataset generation script finished.")
    print(f"Total execution time: {execution_time:.2f} seconds.")
    print(f"===================================================================================")
    logger.info(f"Total execution time: {execution_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Generation Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    configuration = load_config(args.config)
    main(configuration)