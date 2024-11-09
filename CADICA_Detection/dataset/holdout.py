# holdout.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Tuple, List

def run_cleanGroundTruthFileDatasetField(csv_path: str) -> pd.DataFrame:
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
    df = pd.read_csv(csv_path)
    df['GroundTruthFile'] = df['GroundTruthFile'].fillna('nolesion')
    return df

def run_filterByLabels(df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
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
    return df[(df['Lesion'] == False) | (df['LesionLabel'].isin(labels))]

def run_splitData(filtered_df: pd.DataFrame, val_size: float, test_size: float, random_state: int = 39) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training, validation, and testing sets based on unique patients.
    
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
    # Get unique patients
    patients = filtered_df['Patient'].unique()
    
    # For each patient, get the labels associated with them
    patient_labels = {}
    for patient in patients:
        labels = filtered_df[filtered_df['Patient'] == patient]['LesionLabel']
        # Get the most frequent label for the patient
        primary_label = labels.value_counts().idxmax()
        patient_labels[patient] = primary_label
    
    # Create a DataFrame with patients and their primary label
    patient_df = pd.DataFrame(list(patient_labels.items()), columns=['Patient', 'PrimaryLabel'])
    
    # Now, split patients into train, val, test sets, stratifying by primary label
    from sklearn.model_selection import train_test_split
    
    # First split into train_val and test
    train_val_patients, test_patients, train_val_labels, test_labels = train_test_split(
        patient_df['Patient'], patient_df['PrimaryLabel'], test_size=test_size, 
        random_state=random_state, stratify=patient_df['PrimaryLabel']
    )
    
    # Then split train_val into train and val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
    train_patients, val_patients, train_labels, val_labels = train_test_split(
        train_val_patients, train_val_labels, test_size=val_size_adjusted,
        random_state=random_state, stratify=train_val_labels
    )
    
    # Now, select data for each set
    train_df = filtered_df[filtered_df['Patient'].isin(train_patients)]
    val_df = filtered_df[filtered_df['Patient'].isin(val_patients)]
    test_df = filtered_df[filtered_df['Patient'].isin(test_patients)]
    
    return train_df, val_df, test_df

def run_saveSplit(df: pd.DataFrame, output_path: str, split_name: str) -> None:
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
    os.makedirs(output_path, exist_ok=True)
    tqdm.pandas(desc=f'Processing {split_name} split', colour='green')
    
    split_df = df[['SelectedFramesNonLesionVideo', 'SelectedFramesLesionVideo', 'GroundTruthFile', 'Lesion', 'LesionLabel']].copy()
    
    # Determine frame path based on lesion presence
    split_df['Frame_path'] = split_df.progress_apply(
        lambda row: row['SelectedFramesLesionVideo'] if row['Lesion'] else row['SelectedFramesNonLesionVideo'], axis=1
    )
    
    # Determine ground truth path based on existence of ground truth file
    split_df['Groundtruth_path'] = split_df['GroundTruthFile'].apply(
        lambda x: x if isinstance(x, str) and x.startswith('/home/') else 'nolesion'
    )
    
    # Fill missing lesion labels
    split_df['LesionLabel'] = split_df['LesionLabel'].fillna('nolesion')
    
    # Select final columns for output
    split_df = split_df[['LesionLabel', 'Frame_path', 'Groundtruth_path']]
    
    # Save the split to a CSV file
    split_df.to_csv(os.path.join(output_path, f"{split_name}.csv"), index=False)
