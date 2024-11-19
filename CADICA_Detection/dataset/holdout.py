# holdout.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Tuple, List
from collections import Counter
import ast
import numpy as np

def run_cleanGroundTruthFileDatasetField(csv_path: str) -> pd.DataFrame:
    """
    Loads the dataset CSV and fills missing 'GroundTruthFile' values with 'nolesion'.
    """
    df = pd.read_csv(csv_path)
    df['GroundTruthFile'] = df['GroundTruthFile'].fillna('nolesion')
    
    # Parse 'LesionLabel' strings into lists
    df['LesionLabel'] = df['LesionLabel'].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) and x != 'nan' else []
    )
    return df


def run_filterByLabels(df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
    """
    Filters the dataset to retain only samples without lesions or those with specified lesion labels.
    """
    labels_set = set(labels)
    df_filtered = df[(df['Lesion'] == False) | (df['LesionLabel'].apply(lambda x: bool(set(x) & labels_set)))]
    return df_filtered


def run_splitData(filtered_df: pd.DataFrame, val_size: float, test_size: float, random_state: int = 39) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training, validation, and testing sets based on unique patients,
    ensuring that each lesion label is represented in each split according to the desired ratios.
    """
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    
    # Get unique patients
    patients = filtered_df['Patient'].unique()
    
    # Create a patient-label mapping
    patient_label_dict = {}
    for patient in patients:
        patient_df = filtered_df[filtered_df['Patient'] == patient]
        labels_set = set()
        for labels in patient_df['LesionLabel']:
            labels_set.update(labels)
        if not labels_set:
            labels_set.add('nolesion')
        patient_label_dict[patient] = labels_set
    
    # Get all unique labels
    all_labels = set()
    for labels in patient_label_dict.values():
        all_labels.update(labels)
    all_labels = sorted(all_labels)
    
    # Create a binary label matrix for patients
    patient_label_df = pd.DataFrame(0, index=patients, columns=all_labels)
    for patient, labels in patient_label_dict.items():
        for label in labels:
            patient_label_df.loc[patient, label] = 1
    
    # Prepare data for stratification
    X = np.arange(len(patients)).reshape(-1, 1)
    y = patient_label_df.values
    
    # First split: train_val vs test
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_indices, test_indices = next(msss.split(X, y))
    
    # Second split: train vs val
    train_val_X = X[train_val_indices]
    train_val_y = y[train_val_indices]
    
    val_size_adjusted = val_size / (1.0 - test_size)  # Adjust validation size
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_state)
    train_indices, val_indices = next(msss_val.split(train_val_X, train_val_y))
    
    # Get patient IDs for each split
    patients_array = patients
    train_patients = patients_array[train_val_indices[train_indices].flatten()]
    val_patients = patients_array[train_val_indices[val_indices].flatten()]
    test_patients = patients_array[test_indices.flatten()]
    
    # Now, select data for each set
    train_df = filtered_df[filtered_df['Patient'].isin(train_patients)]
    val_df = filtered_df[filtered_df['Patient'].isin(val_patients)]
    test_df = filtered_df[filtered_df['Patient'].isin(test_patients)]
    
    return train_df, val_df, test_df


def run_saveSplit(df: pd.DataFrame, output_path: str, split_name: str) -> None:
    """
    Saves the dataset split to a CSV file.
    """
    os.makedirs(output_path, exist_ok=True)
    tqdm.pandas(desc=f'Processing {split_name} split', colour='green')
    
    split_df = df[['SelectedFramesNonLesionVideo', 'SelectedFramesLesionVideo', 'GroundTruthFile', 'Lesion', 'LesionLabel']].copy()
    
    # Determine frame path based on lesion presence
    split_df['Frame_path'] = split_df.progress_apply(
        lambda row: row['SelectedFramesLesionVideo'] if row['Lesion'] else row['SelectedFramesNonLesionVideo'], axis=1
    )
    
    # Determine ground truth path
    split_df['Groundtruth_path'] = split_df['GroundTruthFile'].apply(
        lambda x: x if x != 'nolesion' else 'nolesion'
    )
    
    # Fill missing lesion labels with ['nolesion']
    split_df['LesionLabel'] = split_df['LesionLabel'].apply(lambda x: x if x else ['nolesion'])
    
    # Convert lesion labels to strings for saving in CSV
    split_df['LesionLabel'] = split_df['LesionLabel'].apply(lambda x: ','.join(x))
    
    # Select final columns for output
    split_df = split_df[['LesionLabel', 'Frame_path', 'Groundtruth_path']]
    
    # Save the split to a CSV file
    split_df.to_csv(os.path.join(output_path, f"{split_name}.csv"), index=False)
