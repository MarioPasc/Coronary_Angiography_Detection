# This script effectively downloads the CADICA dataset and generates a CSV file with path data corresponding to the dataset downloaded.
# This data will be useful to holdout the dataset into train, validation, and test set, which will be useful to perform the careful
# hyperparameter tuning performed in the paper. 
# The output csv will contain the following columns:
# Patient | Video | Frame | Video_Paciente | Lesion | LesionLabel | SelectedFramesNonLesionVideo | SelectedFramesLesionVideo | GroundTruthFile

#############
#  MODULES  #
#############

from CADICA_Detection.dataset.DatasetTools import DatasetTools
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

#############
# VARIABLES #
#############

# Non-configurable variables
DATASET_URL = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/p9bpx9ctcv-2.zip"

# Configurable variables
DESTINATION_FOLDER = "/home/mariopasc/Python/Datasets/try_coronario"  # Change this to the desired destination
OUTPUT_PATH = "/home/mariopasc/Python/Datasets/try_coronario/output.csv"  # Path where the CSV file will be saved
DS_NAME = "CADICA" # Giving the downloaded dataset a custom name


#############
#  SCRIPT   #
#############

# 1. Download Dataset

#DatasetTools.downloadCADICA(destination_folder=DESTINATION_FOLDER, url=DATASET_URL, filename="CADICA_Dataset.zip")

# Rename extracted dataset folder
ORIGINAL_FOLDER = os.path.join(DESTINATION_FOLDER, "CADICA a new dataset for coronary artery disease")
DESTINATION_FOLDER = os.path.join(DESTINATION_FOLDER, DS_NAME)

# Check if the original folder exists and rename it
if os.path.exists(ORIGINAL_FOLDER):
    os.rename(ORIGINAL_FOLDER, DESTINATION_FOLDER)
    print(f"Renamed dataset folder to {DESTINATION_FOLDER}")
else:
    print("Original dataset folder not found, skipping renaming step.")

# 2. Create a .csv file to holdout

# Initialize lists to store data for DataFrame
patient_list = []
video_list = []
frame_list = []
selected_frames_non_lesion_video_list = []
selected_frames_lesion_video_list = []
groundtruth_file_list = []
lesion_list = []
lesion_label_list = []

# Define paths for patients and selected videos
selected_videos_path = os.path.join(DESTINATION_FOLDER, "CADICA", "selectedVideos")  
patients_folders = np.sort(os.listdir(selected_videos_path))

# Process data for each patient
for patient in tqdm(patients_folders, desc="Processing patients ...", colour="green"):
    patient_path = os.path.join(selected_videos_path, patient)
    video_folders = np.sort(os.listdir(patient_path))
    # Retrieve lists of lesion and non-lesion videos
    lesion_videos = DatasetTools.getFilePaths(patient_path, 'lesionVideos.txt')
    non_lesion_videos = DatasetTools.getFilePaths(patient_path, 'nonlesionVideos.txt')

    for video in lesion_videos + non_lesion_videos:
        video_path = os.path.join(patient_path, video)
        
        # Get selected frames and ground truth mappings
        selected_frames = DatasetTools.getSelectedFrames(video_path, patient, video)
        groundtruth_map = DatasetTools.getGroundtruthFiles(video_path, lesion_videos)

        # Process each selected frame
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
                selected_frames_lesion_video_list.append(frame)
                selected_frames_non_lesion_video_list.append("")
            else:
                selected_frames_lesion_video_list.append("")
                selected_frames_non_lesion_video_list.append(frame)
            groundtruth_file_list.append(groundtruth_file)
            lesion_list.append(has_lesion)
            lesion_label_list.append(lesion_label)

# Create a DataFrame from the collected data
data = {
    'Patient': patient_list,
    'Video': video_list,
    'Frame': frame_list,
    'Video_Paciente': [f"{p}_{v}" for p, v in zip(patient_list, video_list)],
    'Lesion': lesion_list,
    'LesionLabel': lesion_label_list,
    'SelectedFramesNonLesionVideo': selected_frames_non_lesion_video_list,
    'SelectedFramesLesionVideo': selected_frames_lesion_video_list,
    'GroundTruthFile': groundtruth_file_list,
}
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
DatasetTools.saveToCsv(df, OUTPUT_PATH)

print(f"Data downloading complete. CSV file saved at: {OUTPUT_PATH}")

# 3. Holdout the CSV File
