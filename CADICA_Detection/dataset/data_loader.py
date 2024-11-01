# CADICA_Detection/Dataset/data_loader.py

import os
import requests
import zipfile
from typing import Optional, Union, List, Dict
import pandas as pd
from tqdm import tqdm

def run_downloadCADICA(destination_folder: str, 
                        url: str = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/p9bpx9ctcv-2.zip",
                        filename: Optional[str] = "CADICA_Dataset.zip") -> Union[str, os.PathLike]:
    """
    Downloads a dataset zip file from the specified URL and extracts it to the destination folder,
    with a progress bar for tracking download progress. If the downloaded zip file contains
    an additional zip file named CADICA.zip, it will be extracted as well.
    
    Args
    -------------
        destination_folder (str): Path to the folder where the dataset should be extracted.
        url (str): URL of the dataset zip file.
        filename (Optional[str]): Name of the zip file to be saved locally.
        
    Raises
    -------------
        Union[str, os.PathLike]: Destination folder passed by the user.
        Exception: If the download or extraction fails.
    
    References
    -------------
    Jiménez-Partinen, Ariadna; Molina-Cabello, 
    Miguel A.; Thurnhofer-Hemsi, Karl; Palomo, Esteban; 
    Rodríguez-Capitán, Jorge; Molina-Ramos, Ana I.; 
    Jiménez-Navarro, Manuel (2024), 
    “CADICA: a new dataset for coronary artery disease”, 
    Mendeley Data, V2, doi: 10.17632/p9bpx9ctcv.2
    """
    
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    
    # Path to save the downloaded file
    zip_path = os.path.join(destination_folder, filename)
    
    try:
        # Start downloading the file with a progress bar
        print(f"Downloading dataset from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for download errors

        # Get total size in bytes for progress tracking
        total_size = int(response.headers.get('content-length', 0))

        # Write the downloaded file to disk with a progress bar
        with open(zip_path, "wb") as file, tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading", ncols=80
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress_bar.update(len(chunk))
        print(f"\nDataset downloaded to {zip_path}")
        
        # Step 1: Extract the main zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(destination_folder)
        print(f"Main dataset extracted to {destination_folder}")
    
        os.rename(os.path.join(destination_folder, "CADICA a new dataset for coronary artery disease"), 
                  os.path.join(destination_folder, "CADICA"))

        # Step 2: Locate and extract the inner CADICA.zip file
        inner_zip_path = os.path.join(destination_folder, "CADICA", "CADICA.zip")
        if os.path.exists(inner_zip_path):
            with zipfile.ZipFile(inner_zip_path, "r") as inner_zip_ref:
                inner_zip_ref.extractall(os.path.join(destination_folder, "CADICA"))
            print(f"Inner CADICA.zip extracted to {os.path.join(destination_folder, 'CADICA')}")
            
            # Optional: Remove the inner zip file after extraction
            os.remove(inner_zip_path)
            print(f"Temporary inner zip file {inner_zip_path} removed.")
        
        return destination_folder
    except requests.exceptions.RequestException as e:
        print("Error during download:", e)
        raise
    except zipfile.BadZipFile as e:
        print("Error during extraction:", e)
        raise
    finally:
        # Optionally, remove the downloaded zip file to save space
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"Temporary zip file {zip_path} removed.")

def run_getFilePaths(patientPath: str, fileName: str) -> List[str]:
    """
    Reads file paths from a specified file and returns them as a list of strings.
    
    Args
    -------------
    patientPath : str
        The directory path where the file is located.
    fileName : str
        The name of the file to read.
        
    Returns
    -------------
    List[str]
        A list of strings, each representing a file path extracted from the file.
    """
    filePath = os.path.join(patientPath, fileName)
    if os.path.isfile(filePath):
        with open(filePath, 'r') as file:
            return [line.strip() for line in file]
    return []

def run_getSelectedFrames(videoPath: str, patient: str, video: str) -> List[str]:
    """
    Retrieves selected frames for a given patient and video.

    Args
    -------------
    videoPath : str
        The path to the directory where video frames are stored.
    patient : str
        Identifier of the patient associated with the video.
    video : str
        The specific video identifier for the patient.

    Returns
    -------------
    List[str]
        A list of file paths for the selected frames associated with the specified video.
    """
    selectedFramesPath = os.path.join(videoPath, f'{patient}_{video}_selectedFrames.txt')
    selectedFrames = []
    if os.path.isfile(selectedFramesPath):
        with open(selectedFramesPath, 'r') as file:
            selectedFrames = [os.path.join(videoPath, 'input', line.strip() + '.png') for line in file]
    return selectedFrames

def run_getGroundtruthFiles(videoPath: str, lesionVideos: List[str]) -> Dict[str, str]:
    """
    Retrieves ground truth files for a given video and maps them by frame identifier.

    Args
    -------------
    videoPath : str
        The path to the directory containing video files.
    lesionVideos : List[str]
        A list of lesion video identifiers.

    Returns
    -------------
    Dict[str, str]
        A dictionary where each key is a frame identifier, and each value is the path to the ground truth file.
    """
    groundtruthFiles = []
    groundtruthMap = {}
    if os.path.basename(videoPath) in lesionVideos:
        groundtruthPath = os.path.join(videoPath, 'groundtruth')
        if os.path.isdir(groundtruthPath):
            groundtruthFiles = [os.path.join(groundtruthPath, f) for f in os.listdir(groundtruthPath) if f.endswith('.txt')]
            groundtruthMap = {os.path.basename(gtFile).split('.')[0]: gtFile for gtFile in groundtruthFiles}
    return groundtruthMap

def run_extractLesionLabel(groundtruthFile: str) -> Optional[str]:
    """
    Extracts the lesion label from a ground truth file if available.

    Args
    -------------
    groundtruthFile : str
        The path to the ground truth file.

    Returns
    -------------
    Optional[str]
        The lesion label extracted from the file, or None if no label is found.
    """
    if os.path.isfile(groundtruthFile):
        with open(groundtruthFile, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5:
                    return parts[4]
    return None

def run_saveToCsv(df: pd.DataFrame, outputCsvPath: str) -> None:
    """
    Saves the DataFrame to a CSV file at the specified path.

    Args
    -------------
    df : pd.DataFrame
        The DataFrame to save to CSV.
    outputCsvPath : str
        Path where the CSV file will be saved.

    Returns
    -------------
    None
    """
    df.to_csv(outputCsvPath, index=False)