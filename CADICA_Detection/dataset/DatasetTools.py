# CADICA_Detection/Dataset/DatasetTools.py

from .data_loader import (
    run_downloadCADICA,
    run_getFilePaths,
    run_getSelectedFrames,
    run_getGroundtruthFiles,
    run_extractLesionLabel,
    run_saveToCsv
)

from .holdout import (
    run_cleanGroundTruthFileDatasetField,
    run_filterByLabels,
    run_saveSplit,
    run_splitData
)

from .undersampling import (
    run_loadDataSplits,
    run_undersampling
)

from .augmentation import (
    run_augmentData,
    run_loadDataAugmentation
)

# Mostly modules needed for typing 
from typing import Optional, Union, List, Dict, Tuple
import os
import pandas as pd

class DatasetTools:
    """
    A utility class for handling dataset-related tasks, such as downloading, preprocessing,
    and splitting data for analysis and model training.
    """

    # data_loader.py tools

    @staticmethod
    def downloadCADICA(destination_folder: str, 
                   url: str = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/p9bpx9ctcv-2.zip",
                   filename: Optional[str] = "CADICA_Dataset.zip") -> Union[str, os.PathLike]:
        """
        Description
        -------------
        Downloads a dataset zip file from the specified URL and extracts it to the destination folder.
        
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
        return run_downloadCADICA(destination_folder = destination_folder, 
                                  url = url, 
                                  filename = filename)

    @staticmethod
    def getFilePaths(patientPath: str, fileName: str) -> List[str]:
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
        return run_getFilePaths(patientPath = patientPath, 
                                fileName = fileName)
    
    @staticmethod
    def getSelectedFrames(videoPath: str, patient: str, video: str) -> List[str]:
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
        return run_getSelectedFrames(videoPath = videoPath, 
                                     patient = patient, 
                                     video = video)

    @staticmethod
    def getGroundtruthFiles(videoPath: str, lesionVideos: List[str]) -> Dict[str, str]:
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
        return run_getGroundtruthFiles(videoPath = videoPath, 
                                       lesionVideos = lesionVideos)

    @staticmethod
    def extractLesionLabel(groundtruthFile: str) -> Optional[str]:
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
        return run_extractLesionLabel(groundtruthFile = groundtruthFile)

    @staticmethod
    def saveToCsv(df: pd.DataFrame, outputCsvPath: str) -> None:
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
        return run_saveToCsv(df = df, 
                             outputCsvPath = outputCsvPath)
    
    # holdout.py tools

    @staticmethod
    def cleanGroundTruthFileDatasetField(csv_path: str) -> pd.DataFrame:
        """
        Loads the dataset CSV and fills missing 'GroundTruthFile' values with 'nolesion'.
        
        Args
        -------------
        csv_path : str
            Path to the CSV file containing the dataset.
            
        Returns
        -------------
        pd.DataFrame
            A DataFrame loaded from the CSV with missing 'GroundTruthFile' values filled.
        """
        return run_cleanGroundTruthFileDatasetField(csv_path = csv_path)

    @staticmethod
    def filterByLabels(df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
        """
        Filters the dataset to retain only samples without lesions or those with specified lesion labels.

        Args
        -------------
        df : pd.DataFrame
            The original dataset DataFrame.
        labels : List[str]
            A list of lesion labels to keep in the filtered dataset.

        Returns
        -------------
        pd.DataFrame
            A filtered DataFrame containing only the specified lesion labels and non-lesion samples.
        """
        return run_filterByLabels(df = df, labels = labels)

    @staticmethod
    def splitData(filtered_df: pd.DataFrame, val_size: float, test_size: float, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataset into training, validation, and testing sets based on unique video samples.

        Args
        -------------
        filtered_df : pd.DataFrame
            The filtered DataFrame to split into train, validation, and test sets.
        val_size : float
            Proportion of the dataset used for validation in relation to training set.
        test_size : float
            Proportion of the dataset used for testing.
        random_state : int, optional
            Random state for reproducibility. Defaults to 39.

        Returns
        -------------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            DataFrames for training, validation, and testing sets, respectively.
        """
        return run_splitData(filtered_df = filtered_df, 
                             val_size = val_size, 
                             test_size = test_size, 
                             random_state = random_state)

    @staticmethod
    def saveSplit(df: pd.DataFrame, output_path: str, split_name: str) -> None:
        """
        Saves the dataset split to a CSV file.

        Args
        -------------
        df : pd.DataFrame
            DataFrame containing the dataset split.
        output_path : str
            Directory where the split CSV will be saved.
        split_name : str
            Name of the split ('train', 'val', or 'test') used in the output filename.

        Returns
        -------------
        None
        """
        return run_saveSplit(df = df, 
                             output_path = output_path, 
                             split_name = split_name)
    
    # undersampling.py tools

    @staticmethod
    def loadDataSplits(holdout_path: str) -> pd.DataFrame:
        """
        Loads the train, val, and test splits from CSV files and concatenates them into a single DataFrame.
        
        Args
        -------------
        holdout_path : str
            Path to the folder containing train.csv, val.csv, and test.csv files.
        
        Returns
        -------------
        pd.DataFrame
            A concatenated DataFrame containing data from all three splits.
        """
        return run_loadDataSplits(holdout_path = holdout_path)

    @staticmethod
    def undersampling(holdout_path: str, class_dict: Dict[str, list]) -> pd.DataFrame:
        """
        Applies undersampling to the train, val, and test splits based on the specified class distribution.

        Args
        -------------
        holdout_path : str
            Path to the folder containing train.csv, val.csv, and test.csv files.
        class_dict : dict
            Dictionary specifying the percentage of images to ignore in each split for each class.

        Returns
        -------------
        pd.DataFrame
            Concatenated DataFrame of all splits after undersampling.
        """
        return run_undersampling(holdout_path = holdout_path, 
                                 class_dict = class_dict)
    
    # augmentation.py tools

    @staticmethod
    def loadDataAugmentation(train_path: str, val_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads training and validation datasets from CSV files.
        
        Args
        -------------
        train_path : str
            Path to the training dataset CSV.
        val_path : str
            Path to the validation dataset CSV.
        
        Returns
        -------------
        Tuple[pd.DataFrame, pd.DataFrame]
            DataFrames for the training and validation datasets.
        """
        return run_loadDataAugmentation(train_path = train_path, 
                                        val_path = val_path)

    @staticmethod
    def augmentData(train_df: pd.DataFrame, val_df: pd.DataFrame, base_output_path: str,
                    augmented_lesion_images: int, augmented_nolesion_images: int, ignore_top_n: int = 0, seed:int = 42) -> None:
        """
        Augments training and validation datasets by applying various augmentations to both lesion and no-lesion images.
        
        Args
        -------------
        train_df : pd.DataFrame
            DataFrame containing the training dataset.
        val_df : pd.DataFrame
            DataFrame containing the validation dataset.
        base_output_path : str
            Base directory where augmented data will be saved.
        augmented_lesion_images : int
            Number of augmented lesion images to generate.
        augmented_nolesion_images : int
            Number of augmented no-lesion images to generate.
        ignore_top_n : int, optional
            Number of most common lesion classes to ignore for augmentation.
            
        Returns
        -------------
        None
        """
        return run_augmentData(train_df = train_df,
                               val_df = val_df,
                               base_output_path = base_output_path,
                               augmented_lesion_images = augmented_lesion_images,
                               augmented_nolesion_images = augmented_nolesion_images,
                               ignore_top_n = ignore_top_n,
                               seed = seed)

