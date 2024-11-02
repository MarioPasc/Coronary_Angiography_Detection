# CADICA_Detection/dataset/formatting.py

import pandas as pd
import os
import shutil
from tqdm import tqdm
from PIL import Image
from typing import Dict

# Required dataset structure for YOLO
STRUCTURE = {
    'images': ['train', 'val', 'test'],
    'labels': ['train', 'val', 'test']
}

def _createDirectories(dataset_dir: str) -> None:
    """
    Create the directory structure for the YOLO dataset.

    Args
    -------------
    dataset_dir : str
        Path to the output directory where the dataset will be generated.

    Returns
    -------------
    None
    """
    for key, subdirs in STRUCTURE.items():
        for subdir in subdirs:
            path = os.path.join(dataset_dir, key, subdir)
            os.makedirs(path, exist_ok=True)

def _convertBboxFormatToYOLO(bbox: str, img_width: int, img_height: int, 
                        class_mappings:Dict[str, int] = {
                                                        "p0_20": 0,
                                                        "p20_50": 0,
                                                        "p50_70": 0,
                                                        "p70_90": 0,
                                                        "p90_98": 0,
                                                        "p99": 0,
                                                        "p100": 0
                                                    }) -> str:
    """
    Convert bounding box from [x, y, w, h, class] format to YOLO format.

    Args
    -------------
    bbox : str
        Bounding box in the format "x y w h class".
    img_width : int
        Width of the image.
    img_height : int
        Height of the image.

    Returns
    -------------
    str
        Bounding box in YOLO format "class x_center y_center width height".
    """
    x, y, w, h, cls = bbox.split()
    x, y, w, h = int(x), int(y), int(w), int(h)
    cls = class_mappings[cls]
    
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w /= img_width
    h /= img_height
    
    return f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

def _copyFiles(df: pd.DataFrame, split: str, dataset_dir: str) -> None:
    """
    Copy image and label files to their respective directories.

    Args
    -------------
    df : pd.DataFrame
        DataFrame containing the file paths.
    split : str
        The dataset split (train, val, or test).
    dataset_dir : str
        Path to the output directory where the dataset will be generated.

    Returns
    -------------
    None
    """
    for _, row in tqdm(df.iterrows(), desc=f'Generating YOLO {split} Dataset...', colour='green'):
        image_src = row['Frame_path']
        label_src = row['Groundtruth_path']
        
        image_dst = os.path.join(dataset_dir, 'images', split, os.path.basename(image_src))
        label_dst = os.path.join(dataset_dir, 'labels', split, os.path.basename(label_src).replace('.txt', '.txt'))
        
        shutil.copy(image_src, image_dst)
        
        if label_src != 'nolesion':
            try:
                with open(label_src, 'r') as f:
                    bboxes = f.readlines()
                
                img_width, img_height = _getImageDimensions(image_src)
                
                with open(label_dst, 'w') as f:
                    for bbox in bboxes:
                        yolo_bbox = _convertBboxFormatToYOLO(bbox.strip(), img_width, img_height)
                        f.write(yolo_bbox + '\n')
            except FileNotFoundError:
                print(f"Label file not found: {label_src}, skipping.")

def _getImageDimensions(image_path: str) -> tuple:
    """
    Get the dimensions of an image.

    Args
    -------------
    image_path : str
        Path to the image file.

    Returns
    -------------
    tuple
        A tuple (width, height) representing the dimensions of the image.
    """
    with Image.open(image_path) as img:
        return img.size

def run_generateDataset(train_csv: str, val_csv: str, test_csv: str, dataset_dir: str) -> None:
    """
    Generate the YOLO dataset by copying files to the appropriate directories.

    Args
    -------------
    train_csv : str
        Path to the training CSV file.
    val_csv : str
        Path to the validation CSV file.
    test_csv : str
        Path to the test CSV file.
    dataset_dir : str
        Path to the output directory where the dataset will be generated.

    Returns
    -------------
    None
    """
    # Create required directories
    _createDirectories(dataset_dir)
    
    # Load CSV files into DataFrames
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    # Copy files for each split
    _copyFiles(train_df, 'train', dataset_dir)
    _copyFiles(val_df, 'val', dataset_dir)
    _copyFiles(test_df, 'test', dataset_dir)