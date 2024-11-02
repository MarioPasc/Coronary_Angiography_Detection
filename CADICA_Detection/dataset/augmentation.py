# CADICA_Detection/dataset/augmentation.py

import os
import pandas as pd
import numpy as np
import cv2
import random
from typing import Tuple

def _augmentSubset(subset_df: pd.DataFrame, base_output_path: str, dataset_type: str,
                   augmented_lesion_images: int, augmented_nolesion_images: int, ignore_top_n: int = 0) -> None:
    """
    Augments a subset of the dataset (train or validation) by generating augmented lesion and no-lesion images.
    
    Args
    -------------
    subset_df : pd.DataFrame
        DataFrame for the specific subset (train or validation).
    base_output_path : str
        Base directory where augmented data will be saved.
    dataset_type : str
        The type of dataset ('train' or 'val').
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
    lesion_images = subset_df[subset_df['LesionLabel'] != 'nolesion']
    nolesion_images = subset_df[subset_df['LesionLabel'] == 'nolesion']

    lesion_label_counts = lesion_images['LesionLabel'].value_counts()

    if ignore_top_n >= len(lesion_label_counts):
        raise ValueError(f"Cannot ignore {ignore_top_n} classes when there are only {len(lesion_label_counts)} unique lesion classes.")

    classes_to_ignore = lesion_label_counts.nlargest(ignore_top_n).index.tolist()
    lesion_label_counts = lesion_label_counts.drop(classes_to_ignore)

    total_labels = len(lesion_label_counts)
    total_instances = lesion_label_counts.sum()
    weights = {label: (total_instances - count) ** 2 / total_instances ** 2 for label, count in lesion_label_counts.items()}
    
    total_weight = sum(weights.values())
    lesion_augmentation_counts = {label: int(augmented_lesion_images * weight / total_weight) for label, weight in weights.items()}

    augmented_images = []

    for label, count in lesion_augmentation_counts.items():
        label_images = lesion_images[lesion_images['LesionLabel'] == label]
        for _ in range(count):
            row = label_images.sample().iloc[0]
            augmented_images.append(_applyAugmentation(row, dataset_type, base_output_path))

    for _ in range(augmented_nolesion_images):
        row = nolesion_images.sample().iloc[0]
        augmented_images.append(_applyAugmentation(row, dataset_type, base_output_path))

    augmented_df = pd.DataFrame(augmented_images, columns=subset_df.columns)
    augmented_df.to_csv(os.path.join(base_output_path, f'augmented_{dataset_type}.csv'), index=False)

    full_df = pd.concat([subset_df, augmented_df], ignore_index=True)
    full_df.to_csv(os.path.join(base_output_path, f'full_augmented_{dataset_type}.csv'), index=False)

def _applyAugmentation(row: pd.Series, dataset_type: str, base_output_path: str) -> pd.Series:
    """
    Applies a random augmentation (brightness, contrast, translation, or x-ray noise) to an image
    and saves the augmented image with a new path.

    Args
    -------------
    row : pd.Series
        A row from the dataset containing metadata about the image, including the path and lesion label.
    dataset_type : str
        Specifies the dataset type (e.g., 'train' or 'val') for organization in the output directory.
    base_output_path : str
        Base directory where the augmented image will be saved.

    Returns
    -------------
    pd.Series
        A new row with the updated path of the augmented image.
    """
    img_path = row['Frame_path']
    img = cv2.imread(img_path)
    aug_type = random.choices(
        ['brightness', 'contrast', 'translation', 'xray_noise'], 
        [0.3, 0.3, 0.3, 0.1], 
        k=1
    )[0]

    if aug_type == 'brightness':
        img = _randomBrightness(img)
    elif aug_type == 'contrast':
        img = _randomContrast(img)
    elif aug_type == 'translation':
        tx, ty = random.randint(-25, 25), random.randint(-25, 25)
        img = _translateImage(img, tx, ty)
    elif aug_type == 'xray_noise':
        img = _xrayNoise(img)

    new_img_path, new_img_name = _generateNewPath(img_path, dataset_type, aug_type, row['LesionLabel'], base_output_path)
    cv2.imwrite(new_img_path, img)

    new_row = row.copy()
    new_row['Frame_path'] = new_img_path
    return new_row

def _randomBrightness(img: np.ndarray) -> np.ndarray:
    """
    Randomly adjusts the brightness of an image by modifying its HSV value channel.

    Args
    -------------
    img : np.ndarray
        The input image in BGR format.

    Returns
    -------------
    np.ndarray
        The brightness-adjusted image in BGR format.
    """
    value = random.uniform(0.7, 1.35)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], value)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def _randomContrast(img: np.ndarray) -> np.ndarray:
    """
    Randomly adjusts the contrast of an image by applying a weighted sum with a random factor.

    Args
    -------------
    img : np.ndarray
        The input image in BGR format.

    Returns
    -------------
    np.ndarray
        The contrast-adjusted image in BGR format.
    """
    alpha = random.uniform(0.75, 1.20)
    return cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, 0)

def _translateImage(img: np.ndarray, tx: int, ty: int) -> np.ndarray:
    """
    Applies a translation to an image by adding reflected borders to prevent black borders from appearing.

    Args
    -------------
    img : np.ndarray
        The input image in BGR format.
    tx : int
        Translation along the x-axis (positive is right, negative is left).
    ty : int
        Translation along the y-axis (positive is down, negative is up).

    Returns
    -------------
    np.ndarray
        The translated image in BGR format, cropped to its original dimensions.
    """
    top, bottom = max(0, ty), max(0, -ty)
    left, right = max(0, tx), max(0, -tx)
    image_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image_with_border = cv2.warpAffine(image_with_border, translation_matrix, (img.shape[1] + left + right, img.shape[0] + top + bottom))
    return translated_image_with_border[top:top + img.shape[0], left:left + img.shape[1]]

def _xrayNoise(img: np.ndarray) -> np.ndarray:
    """
    Adds mild Gaussian noise to an image to simulate x-ray scanner noise.

    Args
    -------------
    img : np.ndarray
        The input image in BGR format.

    Returns
    -------------
    np.ndarray
        The image with added Gaussian noise, clipped to the 0-255 range.
    """
    mean, sigma = 0, 10
    gauss = np.random.normal(mean, sigma, img.shape).astype(np.uint8)
    return np.clip(img + gauss, 0, 255)

def _generateNewPath(img_path: str, dataset_type: str, augmentation_type: str, lesion_label: str, base_output_path: str) -> Tuple[str, str]:
    """
    Generates a unique file path for saving an augmented image based on the dataset type, 
    augmentation type, and lesion label, ensuring no filename conflicts.

    Args
    -------------
    img_path : str
        Original path of the image to be augmented.
    dataset_type : str
        Specifies the dataset type (e.g., 'train' or 'val').
    augmentation_type : str
        The type of augmentation applied (e.g., 'brightness', 'contrast').
    lesion_label : str
        The lesion label associated with the image, used to determine subdirectories.
    base_output_path : str
        Base directory where the augmented image will be saved.

    Returns
    -------------
    Tuple[str, str]
        The new path for the augmented image and the new filename.
    """
    path_parts = img_path.split('/')
    patient_video = path_parts[-4] + "_" + path_parts[-3]
    sub_dir = 'lesion' if lesion_label != 'nolesion' else 'nolesion'
    base_dir = os.path.join(base_output_path, dataset_type, 'images', sub_dir, patient_video)
    os.makedirs(base_dir, exist_ok=True)

    img_name = os.path.basename(img_path).replace('.png', f'_{augmentation_type}')
    img_base_name = os.path.splitext(img_name)[0]
    counter = 1
    new_img_name = f"{img_base_name}_{counter}.png"
    new_img_path = os.path.join(base_dir, new_img_name)

    while os.path.exists(new_img_path):
        counter += 1
        new_img_name = f"{img_base_name}_{counter}.png"
        new_img_path = os.path.join(base_dir, new_img_name)

    return new_img_path, new_img_name

def run_loadDataAugmentation(train_path: str, val_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    return train_df, val_df

def run_augmentData(train_df: pd.DataFrame, val_df: pd.DataFrame, base_output_path: str,
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
    random.seed(seed)
    np.random.seed(seed)
    _augmentSubset(train_df, base_output_path, 'train', augmented_lesion_images, augmented_nolesion_images, ignore_top_n)
    _augmentSubset(val_df, base_output_path, 'val', augmented_lesion_images, augmented_nolesion_images, ignore_top_n)
