# CADICA_Detection/dataset/augmentation.py

import os
import pandas as pd
import numpy as np
import cv2
import random
from typing import Tuple
from tqdm import tqdm
import shutil
from typing import List

def _translateImageOnly(img: np.ndarray, tx: int, ty: int) -> np.ndarray:
    """
    Applies a random translation to an image without modifying bounding boxes.
    Borders are added to avoid black edges when the image is shifted.

    Args:
    -----
    img : np.ndarray
        The input image in BGR format.
    tx : int
        Translation along the x-axis (positive is right, negative is left).
    ty : int
        Translation along the y-axis (positive is down, negative is up).

    Returns:
    --------
    np.ndarray
        Translated image in BGR format with the original dimensions.
    """
    top, bottom = max(0, ty), max(0, -ty)
    left, right = max(0, tx), max(0, -tx)
    image_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image_with_border = cv2.warpAffine(image_with_border, translation_matrix, (img.shape[1] + left + right, img.shape[0] + top + bottom))
    translated_image = translated_image_with_border[top:top + img.shape[0], left:left + img.shape[1]]
    return translated_image

def _augmentSubset(subset_df: pd.DataFrame, base_output_path: str, dataset_type: str,
                   augmented_lesion_images: int, augmented_nolesion_images: int, ignore_top_n: int = 0) -> None:
    """
    Augments a subset of the dataset (train or validation) by applying transformations
    to both lesion and no-lesion images, while ignoring the top N most frequent lesion classes.

    Args:
    -----
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

    Returns:
    --------
    None
    """
    lesion_images = subset_df[subset_df['LesionLabel'] != 'nolesion']
    nolesion_images = subset_df[subset_df['LesionLabel'] == 'nolesion']
    lesion_label_counts = lesion_images['LesionLabel'].value_counts()
    if ignore_top_n >= len(lesion_label_counts):
        raise ValueError(f"Cannot ignore {ignore_top_n} classes when there are only {len(lesion_label_counts)} unique lesion classes.")
    
    classes_to_ignore = lesion_label_counts.nlargest(ignore_top_n).index.tolist()
    lesion_label_counts = lesion_label_counts.drop(classes_to_ignore)
    total_instances = lesion_label_counts.sum()
    weights = {label: (total_instances - count) ** 2 / total_instances ** 2 for label, count in lesion_label_counts.items()}
    total_weight = sum(weights.values())
    lesion_augmentation_counts = {label: int(augmented_lesion_images * weight / total_weight) for label, weight in weights.items()}

    augmented_images = []
    for label, count in tqdm(lesion_augmentation_counts.items(), desc="Applying data augmentation to lesion images ...", colour='green'):
        label_images = lesion_images[lesion_images['LesionLabel'] == label]
        for _ in range(count):
            row = label_images.sample().iloc[0]
            augmented_images.append(_applyAugmentation(row, dataset_type, base_output_path))

    for _ in tqdm(range(augmented_nolesion_images), desc='Applying data augmentation to non-lesion images ...', colour='green'):
        row = nolesion_images.sample().iloc[0]
        augmented_images.append(_applyAugmentation(row, dataset_type, base_output_path))

    augmented_df = pd.DataFrame(augmented_images, columns=subset_df.columns)
    augmented_df.to_csv(os.path.join(base_output_path, f'augmented_{dataset_type}.csv'), index=False)
    full_df = pd.concat([subset_df, augmented_df], ignore_index=True)
    full_df.to_csv(os.path.join(base_output_path, f'processed_{dataset_type}.csv'), index=False)

def _applyAugmentation(row: pd.Series, dataset_type: str, base_output_path: str) -> pd.Series:
    """
    Applies a random augmentation to an image and updates the bounding box if applicable.

    Args:
    -----
    row : pd.Series
        A row from the dataset containing metadata about the image, including the path and lesion label.
    dataset_type : str
        Specifies the dataset type (e.g., 'train' or 'val') for organization in the output directory.
    base_output_path : str
        Base directory where the augmented image will be saved.

    Returns:
    --------
    pd.Series
        A new row with the updated path of the augmented image and bounding box information if applicable.
    """
    img_path = row['Frame_path']
    img = cv2.imread(img_path)
    aug_type = random.choices(['brightness', 'contrast', 'translation', 'xray_noise'], [0.3, 0.3, 0.3, 0.1], k=1)[0]
    groundtruth_path = row['Groundtruth_path']
    augmentation_type = ""

    if aug_type == 'brightness':
        img = _randomBrightness(img)
        augmentation_type = "brightness"
    elif aug_type == 'contrast':
        img = _randomContrast(img)
        augmentation_type = "contrast"
    elif aug_type == 'translation':
        tx, ty = random.randint(-25, 25), random.randint(-25, 25)
        if groundtruth_path != 'nolesion':
            img, bbox_coords = _randomTranslation(img, groundtruth_path, tx, ty)
            augmentation_type = "translation"
        else:
            img = _translateImageOnly(img, tx, ty)
            augmentation_type = "translation"
    elif aug_type == 'xray_noise':
        img = _xrayNoise(img)
        augmentation_type = "xray_noise"

    new_img_path, new_img_name = _generateNewPath(img_path, dataset_type, augmentation_type, row['LesionLabel'], base_output_path)
    cv2.imwrite(new_img_path, img)

    new_gt_path = 'nolesion'
    if groundtruth_path != 'nolesion':
        if augmentation_type == "translation":
            new_gt_path = _saveTranslatedBoundingBox(bbox_coords, new_img_name, dataset_type, base_output_path)
        else:
            new_gt_path = _copyBoundingBox(groundtruth_path, new_img_name, dataset_type, base_output_path)

    new_row = row.copy()
    new_row['Frame_path'] = new_img_path
    new_row['Groundtruth_path'] = new_gt_path
    return new_row

def _randomTranslation(img: np.ndarray, gt_path: str, tx: int, ty: int) -> Tuple[np.ndarray, List[int]]:
    """
    Applies a random translation to an image with a bounding box, updating the bounding box coordinates.

    Args:
    -----
    img : np.ndarray
        The input image in BGR format.
    gt_path : str
        Path to the ground truth bounding box file.
    tx : int
        Translation along the x-axis.
    ty : int
        Translation along the y-axis.

    Returns:
    --------
    Tuple[np.ndarray, List[int]]
        The translated image and the new bounding box coordinates.
    """
    with open(gt_path, 'r') as f:
        bbox = f.readline().strip().split()
        x, y, w, h, label = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), str(bbox[4])

    top, bottom = max(0, ty), max(0, -ty)
    left, right = max(0, tx), max(0, -tx)
    image_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image_with_border = cv2.warpAffine(image_with_border, translation_matrix, (img.shape[1] + left + right, img.shape[0] + top + bottom))
    translated_image = translated_image_with_border[top:top + img.shape[0], left:left + img.shape[1]]

    new_x, new_y = x + tx, y + ty
    return translated_image, [new_x, new_y, w, h, label]

def _saveTranslatedBoundingBox(bbox_coords: List[int], new_img_name: str, dataset_type: str, base_output_path: str) -> str:
    """
    Saves a translated bounding box to a new file in the output directory.

    Args:
    -----
    bbox_coords : List[int]
        Coordinates of the translated bounding box [x, y, width, height, label].
    new_img_name : str
        The name of the new image file.
    dataset_type : str
        The dataset type ('train' or 'val') for directory organization.
    base_output_path : str
        Base directory where the bounding box will be saved.

    Returns:
    --------
    str
        Path to the new bounding box file.
    """
    path_parts = new_img_name.split('_')
    patient_video = path_parts[0] + '_' + path_parts[1]
    new_dir = os.path.join(base_output_path, dataset_type, 'labels', 'lesion', patient_video)
    os.makedirs(new_dir, exist_ok=True)
    gt_name = new_img_name.replace('.png', '.txt')
    new_gt_path = os.path.join(new_dir, gt_name)
    with open(new_gt_path, 'w') as f:
        f.write(f"{bbox_coords[0]} {bbox_coords[1]} {bbox_coords[2]} {bbox_coords[3]} {bbox_coords[4]}")
    return new_gt_path

def _copyBoundingBox(gt_path: str, new_img_name: str, dataset_type: str, base_output_path: str) -> str:
    """
    Copies an existing bounding box file to the specified output directory.

    Args:
    -----
    gt_path : str
        Original path of the bounding box file.
    new_img_name : str
        Name of the augmented image file.
    dataset_type : str
        Dataset type ('train' or 'val').
    base_output_path : str
        Base directory where the bounding box file will be copied.

    Returns:
    --------
    str
        Path to the copied bounding box file.
    """
    path_parts = new_img_name.split('_')
    patient_video = path_parts[0] + '_' + path_parts[1]
    new_dir = os.path.join(base_output_path, dataset_type, 'labels', 'lesion', patient_video)
    os.makedirs(new_dir, exist_ok=True)
    gt_name = new_img_name.replace('.png', '.txt')
    new_gt_path = os.path.join(new_dir, gt_name)
    shutil.copy(gt_path, new_gt_path)
    return new_gt_path

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
    print("Applying augmentation to train set.")
    _augmentSubset(train_df, base_output_path, 'train', augmented_lesion_images, augmented_nolesion_images, ignore_top_n)
    print("Applying augmentation to validation set.")
    _augmentSubset(val_df, base_output_path, 'val', augmented_lesion_images, augmented_nolesion_images, ignore_top_n)
