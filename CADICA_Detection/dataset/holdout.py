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
    
    Ensures that labels with small counts are included in validation and test sets.
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
    
    # Get counts of patients per label
    patient_label_counts = patient_df.groupby('PrimaryLabel').size()
    
    # Identify labels with small counts (less than 3 patients)
    small_labels = patient_label_counts[patient_label_counts < 3].index.tolist()
    
    # Get patients with small labels
    small_label_patients_df = patient_df[patient_df['PrimaryLabel'].isin(small_labels)]
    small_patients = small_label_patients_df['Patient'].tolist()
    
    # Assign patients with small labels to validation and test sets
    val_patients_small = []
    test_patients_small = []
    train_patients_small = []
    
    for label in small_labels:
        patients_with_label = small_label_patients_df[small_label_patients_df['PrimaryLabel'] == label]['Patient'].tolist()
        num_patients = len(patients_with_label)
        if num_patients >= 3:
            # Assign one patient to each set
            val_patients_small.append(patients_with_label[0])
            test_patients_small.append(patients_with_label[1])
            train_patients_small.extend(patients_with_label[2:])
        elif num_patients == 2:
            # Assign one patient to val, one to test
            val_patients_small.append(patients_with_label[0])
            test_patients_small.append(patients_with_label[1])
        elif num_patients ==1:
            # Assign patient to validation
            val_patients_small.append(patients_with_label[0])
    
    # Remove small label patients from patient_df
    remaining_patient_df = patient_df[~patient_df['Patient'].isin(small_patients)]
    
    # Now, split remaining patients into train_val and test
    from sklearn.model_selection import train_test_split
    
    train_val_patients, test_patients_remaining, train_val_labels, test_labels = train_test_split(
        remaining_patient_df['Patient'], remaining_patient_df['PrimaryLabel'], test_size=test_size, 
        random_state=random_state, stratify=remaining_patient_df['PrimaryLabel']
    )
    
    # Then split train_val into train and val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
    train_patients_remaining, val_patients_remaining, train_labels, val_labels = train_test_split(
        train_val_patients, train_val_labels, test_size=val_size_adjusted,
        random_state=random_state, stratify=train_val_labels
    )
    
    # Combine patients
    train_patients = train_patients_remaining.tolist() + train_patients_small
    val_patients = val_patients_remaining.tolist() + val_patients_small
    test_patients = test_patients_remaining.tolist() + test_patients_small
    
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
