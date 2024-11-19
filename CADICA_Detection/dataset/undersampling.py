# CADICA_Detection/dataset/undersampling.py 

import pandas as pd
import os
from typing import Dict, Tuple, List

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

def _undersampleClass(df: pd.DataFrame, class_dict: Dict[str, list], split_type: str, freezed_classes: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies undersampling to the DataFrame for the specified split type (train, val, or test).

    Args
    -----
    df : pd.DataFrame
        DataFrame for the specific split (train, val, test).
    class_dict : dict
        Dictionary where keys are class names and values are lists with 3 integers representing
        the percentage of images to ignore in [train, val, test] splits.
    split_type : str
        The type of split ('train', 'val', 'test').
    freezed_classes : List[str]
        List of class names that should not be undersampled.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The DataFrame after applying undersampling and a DataFrame containing the rows that were deleted.
    """
    import ast
    from collections import defaultdict

    # Parse 'LesionLabel' into lists
    df['LesionLabelList'] = df['LesionLabel'].apply(
        lambda x: x.split(',') if pd.notnull(x) and x != 'nolesion' else []
    )

    split_index_map = {'train': 0, 'val': 1, 'test': 2}
    split_index = split_index_map[split_type]
    deleted_rows = pd.DataFrame(columns=df.columns)  # Initialize DataFrame to store deleted rows

    images_to_remove = set()

    # Collect images eligible for removal per class
    for class_name, percentages in class_dict.items():
        percentage_to_ignore = percentages[split_index]
        if percentage_to_ignore > 0:
            # Get images that contain the class and do not contain any freezed classes
            class_df = df[df['LesionLabelList'].apply(
                lambda labels: (class_name in labels) and not any(freeze_class in labels for freeze_class in freezed_classes)
            )]
            num_images_with_class = len(class_df)
            num_to_remove = int(num_images_with_class * (percentage_to_ignore / 100))

            # Get indices of images not already selected for removal
            eligible_indices = list(set(class_df.index) - images_to_remove)

            # If there are not enough images left, adjust num_to_remove
            num_to_remove = min(num_to_remove, len(eligible_indices))

            if num_to_remove > 0:
                # Randomly select images to remove
                rows_to_remove_indices = class_df.loc[eligible_indices].sample(n=num_to_remove, random_state=42).index
                images_to_remove.update(rows_to_remove_indices)

    # Convert set to list for indexing
    images_to_remove_list = list(images_to_remove)

    # Remove selected images from df
    deleted_rows = df.loc[images_to_remove_list]
    df = df.drop(index=images_to_remove_list)

    # Clean up 'LesionLabelList' column
    df = df.drop(columns=['LesionLabelList'])
    deleted_rows = deleted_rows.drop(columns=['LesionLabelList'])

    return df, deleted_rows

def run_undersampling(holdout_path: str, class_dict: Dict[str, list], deleted_images_path: str, freezed_classes: List[str]) -> pd.DataFrame:
    """
    Applies undersampling to the train, val, and test splits based on the specified class distribution.
    Saves the modified splits and a log of deleted images to a specified path.

    Args
    -----
    holdout_path : str
        Path to the folder containing train.csv, val.csv, and test.csv files.
    class_dict : dict
        Dictionary specifying the percentage of images to ignore in each split for each class.
    deleted_images_path : str
        Path where the CSV file of deleted images will be saved.
    freezed_classes : List[str]
        List of class names that should not be undersampled.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all splits after undersampling.
    """
    train_csv = os.path.join(holdout_path, 'train.csv')
    val_csv = os.path.join(holdout_path, 'val.csv')
    test_csv = os.path.join(holdout_path, 'test.csv')

    # Apply undersampling to each split and collect deleted images
    train_df, train_deleted = _undersampleClass(pd.read_csv(train_csv), class_dict, split_type='train', freezed_classes=freezed_classes)
    val_df, val_deleted = _undersampleClass(pd.read_csv(val_csv), class_dict, split_type='val', freezed_classes=freezed_classes)
    test_df, test_deleted = _undersampleClass(pd.read_csv(test_csv), class_dict, split_type='test', freezed_classes=freezed_classes)

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