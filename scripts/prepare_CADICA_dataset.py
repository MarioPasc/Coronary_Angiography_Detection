#!/usr/bin/env python3

# This script contains all the functionality necessary to generate the input dataset used to train the YOLOv8 Ultralytic's model
# used for generating the results of the paper
# This script uses the tools provided by the class DatasetTools, which serves as an entry point to the dataset submodule from the CADICA_Detection package.
# The script follows these steps:
#   1. Dataset downloading
#   2. Dataset analysis: Generate a .csv file with all the paths and lesion tags for the images.
#   3. Holdout: Generate 3 .csv files with the paths and information about the images for each set: train/val/test.
#   4. Data undersampling: Undersample specific lesion classes from the original dataset to avoid introducing a bias into the model.
#   5. Data augmentation: Using our own augmentation tools, recreate different scenarios that could take place when adquiring the images to compensate
#      the lower amount of samples for some lesion classes, effectively creating new images for the model to train on, and balancing the target class.
#   6. YOLO dataset formatting: Now that we have all the new images generated and our splits done, format the dataset in order to feed it to the YOLO model.
#
# Considerations about the code:
# - The default setting of this script generate the dataset used for the paper.
# - The test set did not undergo any agumentation/undersampling step.

# export PYTHONPATH="${PYTHONPATH}:/home/mariopasc/Python/Projects/Coronary_Angiography_Detection"

CONFIG_PATH = "./scripts/config.yaml"

import os
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from CADICA_Detection.dataset.DatasetTools import DatasetTools
import yaml

# Set up logging
os.makedirs('./logs', exist_ok=True)
logging.basicConfig(filename='./logs/dataset_generation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting dataset generation process.")

def load_config(yaml_path: str) -> dict:
    """
    Loads configuration parameters from a YAML file into a dictionary.
    
    Args
    -----
    yaml_path : str
        Path to the YAML configuration file.
    
    Returns
    -----
    dict
        Configuration parameters loaded from YAML.
    """
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error("Error loading config.yaml file; Are you executing the script from the root folder? If so, check this .py and change CONFIG_PATH.")

# Load the CONFIG variable
CONFIG = load_config(CONFIG_PATH)

# 1. Download and Prepare Dataset
def download_and_prepare_dataset():
    try:
        logging.info("Step 1: Starting dataset download and preparation.")
        destination_folder = CONFIG["DESTINATION_FOLDER"]
        
        # Download dataset
        DatasetTools.downloadCADICA(destination_folder=destination_folder, url=CONFIG["DATASET_URL"], filename="CADICA_Dataset.zip")
        
        # Rename extracted folder
        original_folder = os.path.join(destination_folder, "CADICA a new dataset for coronary artery disease")
        renamed_folder = os.path.join(destination_folder, CONFIG["DS_NAME"])
        
        if os.path.exists(original_folder):
            os.rename(original_folder, renamed_folder)
            logging.info(f"Renamed dataset folder to {renamed_folder}")
        else:
            logging.warning("Original dataset folder not found. Skipping renaming step.")
        
        logging.info("Step 1 complete: Dataset downloaded and prepared.")
    except Exception as e:
        logging.error(f"Error in Step 1: Dataset download and preparation failed - {e}")

# 2. Generate Dataset Analysis CSV
def generate_dataset_analysis():
    try:
        logging.info("Step 2: Generating dataset analysis CSV.")
        
        # Define paths
        destination_folder = os.path.join(CONFIG["DESTINATION_FOLDER"], CONFIG["DS_NAME"])
        selected_videos_path = os.path.join(destination_folder, "CADICA", "selectedVideos")
        output_path = CONFIG["OUTPUT_PATH"]
        data_csv_name = CONFIG["DATA_CSV_NAME"]
        
        # Initialize lists for DataFrame
        patient_list, video_list, frame_list = [], [], []
        selected_frames_non_lesion, selected_frames_lesion = [], []
        groundtruth_files, lesion_list, lesion_labels = [], [], []

        # Process each patient
        patients_folders = np.sort(os.listdir(selected_videos_path))
        for patient in tqdm(patients_folders, desc="Processing patients...", colour="green"):
            patient_path = os.path.join(selected_videos_path, patient)
            lesion_videos = DatasetTools.getFilePaths(patient_path, 'lesionVideos.txt')
            non_lesion_videos = DatasetTools.getFilePaths(patient_path, 'nonlesionVideos.txt')

            for video in lesion_videos + non_lesion_videos:
                video_path = os.path.join(patient_path, video)
                selected_frames = DatasetTools.getSelectedFrames(video_path, patient, video)
                groundtruth_map = DatasetTools.getGroundtruthFiles(video_path, lesion_videos)

                for frame in selected_frames:
                    frame_id = os.path.basename(frame).split('_')[-1].split('.')[0]
                    groundtruth_file = groundtruth_map.get(f'{patient}_{video}_{frame_id}', '')
                    has_lesion = groundtruth_file != ''
                    lesion_label = DatasetTools.extractLesionLabel(groundtruth_file) if has_lesion else None

                    # Append data to lists
                    patient_list.append(patient)
                    video_list.append(video)
                    frame_list.append(frame_id)
                    if video in lesion_videos:
                        selected_frames_lesion.append(frame)
                        selected_frames_non_lesion.append("")
                    else:
                        selected_frames_lesion.append("")
                        selected_frames_non_lesion.append(frame)
                    groundtruth_files.append(groundtruth_file)
                    lesion_list.append(has_lesion)
                    lesion_labels.append(lesion_label)

        # Create DataFrame
        data = {
            'Patient': patient_list,
            'Video': video_list,
            'Frame': frame_list,
            'Video_Paciente': [f"{p}_{v}" for p, v in zip(patient_list, video_list)],
            'Lesion': lesion_list,
            'LesionLabel': lesion_labels,
            'SelectedFramesNonLesionVideo': selected_frames_non_lesion,
            'SelectedFramesLesionVideo': selected_frames_lesion,
            'GroundTruthFile': groundtruth_files,
        }
        dataset_info_df = pd.DataFrame(data)

        # Save to CSV
        csv_path = os.path.join(output_path, data_csv_name)
        DatasetTools.saveToCsv(dataset_info_df, csv_path)
        logging.info(f"Step 2 complete: Dataset analysis CSV saved to {csv_path}")
    
    except Exception as e:
        logging.error(f"Error in Step 2: Dataset analysis CSV generation failed - {e}")

# 3. Split Data for Holdout
def split_holdout_sets():
    try:
        logging.info("Step 3: Generating holdout dataset splits.")
        
        # Load and clean the dataset
        csv_path = os.path.join(CONFIG["OUTPUT_PATH"], CONFIG["DATA_CSV_NAME"])
        df = DatasetTools.cleanGroundTruthFileDatasetField(csv_path)

        # Filter by labels
        filtered_df = DatasetTools.filterByLabels(df, CONFIG["LABELS"])

        # Split the data using the updated function
        train_df, val_df, test_df = DatasetTools.splitData(
            filtered_df, CONFIG["VAL_SIZE"], CONFIG["TEST_SIZE"], CONFIG["RANDOM_SEED"]
        )

        # Save splits to CSV
        holdout_folder = os.path.join(CONFIG["OUTPUT_PATH"], "CADICA_Holdout_Info")
        os.makedirs(holdout_folder, exist_ok=True)
        DatasetTools.saveSplit(train_df, holdout_folder, 'train')
        DatasetTools.saveSplit(val_df, holdout_folder, 'val')
        DatasetTools.saveSplit(test_df, holdout_folder, 'test')

        logging.info(f"Step 3 complete: Holdout splits saved in {holdout_folder}")
    
    except Exception as e:
        logging.error(f"Error in Step 3: Holdout split generation failed - {e}")

# 4. Undersample Overrepresented Lesion Classes
def apply_undersampling():
    try:
        logging.info("Step 4: Applying undersampling to overrepresented lesion classes.")
        
        # Define holdout folder path
        holdout_folder = os.path.join(CONFIG["OUTPUT_PATH"], "CADICA_Holdout_Info")
        
        # Load original data for comparison and apply undersampling
        pre_undersampling_df = DatasetTools.loadDataSplits(holdout_folder)
        post_undersampling_df = DatasetTools.undersampling(holdout_folder, CONFIG["CLASS_UNDERSAMPLING"], CONFIG["OUTPUT_PATH"])

        logging.info(f"Step 4 complete: Undersampling applied. Modified content saved in {holdout_folder}.")
    except Exception as e:
        logging.error(f"Error in Step 4: Undersampling failed - {e}")

# 5. Apply Data Augmentation
def apply_data_augmentation():
    try:
        logging.info("Step 5: Applying data augmentation to balance classes.")
        
        # Define paths for augmentation
        holdout_folder = os.path.join(CONFIG["OUTPUT_PATH"], "CADICA_Undersampled_Images")
        augmentation_path = os.path.join(CONFIG["OUTPUT_PATH"], "CADICA_Augmented_Images")
        
        train_path = os.path.join(holdout_folder, 'processed_train.csv')
        val_path = os.path.join(holdout_folder, 'processed_val.csv')
        
        # Load data for augmentation
        train_df, val_df = DatasetTools.loadDataAugmentation(train_path, val_path)
        
        # Apply augmentation
        DatasetTools.augmentData(
            train_df, val_df, augmentation_path, 
            CONFIG["AUGMENTED_LESION_IMAGES"], CONFIG["AUGMENTED_NONLESION_IMAGES"], 
            CONFIG["IGNORE_TOP_N"], CONFIG["RANDOM_SEED"]
        )

        logging.info(f"Step 5 complete: Data augmentation completed. Results saved in {augmentation_path}.")
    except Exception as e:
        logging.error(f"Error in Step 5: Data augmentation failed - {e}")

# 6. Format the Dataset for YOLO
def format_for_yolo():
    try:
        logging.info("Step 6: Formatting dataset for YOLO.")
        
        # Define paths for YOLO formatting
        augmentation_path = os.path.join(CONFIG["OUTPUT_PATH"], "CADICA_Augmented_Images")
        undersampling_folder = os.path.join(CONFIG["OUTPUT_PATH"], "CADICA_Undersampled_Images")
        
        train_path = os.path.join(augmentation_path, 'processed_train.csv')  # Augmented train data
        val_path = os.path.join(augmentation_path, 'processed_val.csv')      # Augmented val data
        test_path = os.path.join(undersampling_folder, 'processed_test.csv') # Undersampled test split. This test.csv is the original since we did not remove any samples from it
        
        output_folder = os.path.join(CONFIG["DESTINATION_FOLDER"], CONFIG["YOLO_DATASET_FOLDER_NAME"])
        
        # Generate YOLO dataset
        DatasetTools.generateYOLODataset(train_path, val_path, test_path, output_folder)
        
        logging.info(f"Step 6 complete: YOLO dataset saved in {output_folder}.")
    except Exception as e:
        logging.error(f"Error in Step 6: YOLO dataset formatting failed - {e}")

# Main execution
def main():
    # Execute all steps in sequence
    #download_and_prepare_dataset()
    generate_dataset_analysis()
    split_holdout_sets()
    apply_undersampling()
    apply_data_augmentation()
    format_for_yolo()
    
    print("Dataset generation pipeline completed. Check log for details/errors.")

if __name__ == "__main__":
    main()

