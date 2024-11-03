# CADICA_Detection/dataset/undersampling.py 

import pandas as pd
import os
from typing import Dict, Tuple

def run_loadDataSplits(holdout_path: str) -> pd.DataFrame:
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
    train_csv = os.path.join(holdout_path, 'train.csv')
    val_csv = os.path.join(holdout_path, 'val.csv')
    test_csv = os.path.join(holdout_path, 'test.csv')
    
    return pd.concat([
        pd.read_csv(train_csv),
        pd.read_csv(val_csv),
        pd.read_csv(test_csv)
    ], ignore_index=True)

def _undersampleClass(df: pd.DataFrame, class_dict: Dict[str, list], split_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies undersampling to the DataFrame for the specified split type (train, val, or test).

    Args
    -------------
    df : pd.DataFrame
        DataFrame for the specific split (train, val, test).
    class_dict : dict
        Dictionary where keys are class names and values are lists with 3 integers representing
        the percentage of images to ignore in [train, val, test] splits.
    split_type : str
        The type of split ('train', 'val', 'test').

    Returns
    -------------
    Tuple[pd.DataFrame, pd.DataFrame]
        The DataFrame after applying undersampling and a DataFrame containing the rows that were deleted.
    """
    split_index_map = {'train': 0, 'val': 1, 'test': 2}
    split_index = split_index_map[split_type]
    deleted_rows = pd.DataFrame(columns=df.columns)  # Initialize DataFrame to store deleted rows

    for class_name, percentages in class_dict.items():
        percentage_to_ignore = percentages[split_index]
        if percentage_to_ignore > 0:
            class_df = df[df['LesionLabel'] == class_name]
            num_to_remove = int(len(class_df) * (percentage_to_ignore / 100))
            rows_to_remove = class_df.sample(n=num_to_remove, random_state=42)
            deleted_rows = pd.concat([deleted_rows, rows_to_remove])  # Append deleted rows
            df = df.drop(rows_to_remove.index)  # Remove rows from the original DataFrame

    return df, deleted_rows

def run_undersampling(holdout_path: str, class_dict: Dict[str, list], deleted_images_path: str) -> pd.DataFrame:
    """
    Applies undersampling to the train, val, and test splits based on the specified class distribution.
    Saves the modified splits and a log of deleted images to a specified path.

    Args
    -------------
    holdout_path : str
        Path to the folder containing train.csv, val.csv, and test.csv files.
    class_dict : dict
        Dictionary specifying the percentage of images to ignore in each split for each class.
    deleted_images_path : str
        Path where the CSV file of deleted images will be saved.

    Returns
    -------------
    pd.DataFrame
        Concatenated DataFrame of all splits after undersampling.
    """
    train_csv = os.path.join(holdout_path, 'train.csv')
    val_csv = os.path.join(holdout_path, 'val.csv')
    test_csv = os.path.join(holdout_path, 'test.csv')

    # Apply undersampling to each split and collect deleted images
    train_df, train_deleted = _undersampleClass(pd.read_csv(train_csv), class_dict, split_type='train')
    val_df, val_deleted = _undersampleClass(pd.read_csv(val_csv), class_dict, split_type='val')
    test_df, test_deleted = _undersampleClass(pd.read_csv(test_csv), class_dict, split_type='test')

    # Save log for the deleted images
    folder_undersampling = os.path.join(deleted_images_path, "CADICA_Undersampled_Images")
    os.makedirs(folder_undersampling, exist_ok=True)
    
    train_deleted.to_csv(path_or_buf=os.path.join(folder_undersampling, 'undersampled_train.csv'), index=False)
    val_deleted.to_csv(path_or_buf=os.path.join(folder_undersampling, 'undersampled_val.csv'), index=False)
    test_deleted.to_csv(path_or_buf=os.path.join(folder_undersampling, 'undersampled_test.csv'), index=False)

    # Save modified DataFrames
    train_df.to_csv(os.path.join(folder_undersampling, "processed_train.csv"), index=False)
    val_df.to_csv(os.path.join(folder_undersampling, "processed_val.csv"), index=False)
    test_df.to_csv(os.path.join(folder_undersampling, "processed_test.csv"), index=False)

    return pd.concat([train_df, val_df, test_df], ignore_index=True)