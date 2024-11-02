# CADICA_Detection/dataset/undersampling.py 

import pandas as pd
import os
from typing import Dict

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

def _undersampleClass(df: pd.DataFrame, class_dict: Dict[str, list], split_type: str) -> pd.DataFrame:
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
    pd.DataFrame
        The DataFrame after applying undersampling based on the specified percentages.
    """
    split_index_map = {'train': 0, 'val': 1, 'test': 2}
    split_index = split_index_map[split_type]

    for class_name, percentages in class_dict.items():
        percentage_to_ignore = percentages[split_index]
        if percentage_to_ignore > 0:
            class_df = df[df['LesionLabel'] == class_name]
            num_to_remove = int(len(class_df) * (percentage_to_ignore / 100))
            rows_to_remove = class_df.sample(n=num_to_remove, random_state=42).index
            df = df.drop(rows_to_remove)

    return df

def run_undersampling(holdout_path: str, class_dict: Dict[str, list]) -> pd.DataFrame:
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
    train_csv = os.path.join(holdout_path, 'train.csv')
    val_csv = os.path.join(holdout_path, 'val.csv')
    test_csv = os.path.join(holdout_path, 'test.csv')

    # Apply undersampling to each split
    train_df = _undersampleClass(pd.read_csv(train_csv), class_dict, split_type='train')
    val_df = _undersampleClass(pd.read_csv(val_csv), class_dict, split_type='val')
    test_df = _undersampleClass(pd.read_csv(test_csv), class_dict, split_type='test')

    # Save modified DataFrames
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return pd.concat([train_df, val_df, test_df], ignore_index=True)