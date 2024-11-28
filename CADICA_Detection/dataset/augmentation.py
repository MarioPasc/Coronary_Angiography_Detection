# CADICA_Detection/dataset/augmentation.py

# Most references and implemented methods taken from https://www.sciencedirect.com/science/article/pii/S277244252400042X#b141

import os
import pandas as pd
import numpy as np
import cv2
import random
from typing import Tuple, List, Union
from tqdm import tqdm
import shutil
from scipy.ndimage import gaussian_filter


class Augmentor:
    def __init__(self, base_output_path: str,
                 train_augmentation_counts_csv: str,
                 val_augmentation_counts_csv: str,
                 augmented_nolesion_images_train: int,
                 augmented_nolesion_images_val: int,
                 seed: int = 42):
        """
        Initializes the Augmentor with the specified parameters.

        Args:
        -----
        base_output_path : str
            Base directory where augmented data will be saved.
        train_augmentation_counts_csv : str
            Path to the CSV file containing augmentation counts per label for the train set.
        val_augmentation_counts_csv : str
            Path to the CSV file containing augmentation counts per label for the validation set.
        augmented_nolesion_images_train : int
            Number of augmented no-lesion images to generate for the train set.
        augmented_nolesion_images_val : int
            Number of augmented no-lesion images to generate for the validation set.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.base_output_path = base_output_path
        self.augmented_nolesion_images = {
            'train': augmented_nolesion_images_train,
            'val': augmented_nolesion_images_val
        }
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Load augmentation counts DataFrames for both train and val
        self.augmentation_counts = {
            'train': pd.read_csv(train_augmentation_counts_csv),
            'val': pd.read_csv(val_augmentation_counts_csv)
        }

        # Create dictionaries for quick access
        self.label_aug_counts = {
            'train': dict(zip(self.augmentation_counts['train']['Label'],
                              self.augmentation_counts['train']['Augmented_Counts'])),
            'val': dict(zip(self.augmentation_counts['val']['Label'],
                            self.augmentation_counts['val']['Augmented_Counts']))
        }

        # Mapping of augmentation types to their functions and whether they modify bounding boxes
        self.augmentation_functions = {
            'brightness': (self._random_brightness, False),
            'contrast': (self._random_contrast, False),
            'translation': (self._random_translation, True),
            'xray_noise': (self._xray_noise, False),
            'elastic_deformation': (self._elastic_deformation, True),
        }

        # Probabilities for each augmentation type
        self.augmentation_weights = [0.33, 0.33, 0.33, 0, 0]

    def augment_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """
        Coordinates the augmentation process for training and validation datasets.

        Args:
        -----
        train_df : pd.DataFrame
            DataFrame containing the training dataset.
        val_df : pd.DataFrame
            DataFrame containing the validation dataset.
        """
        print("Applying augmentation to train set.")
        self._augment_subset(train_df, 'train')
        print("Applying augmentation to validation set.")
        self._augment_subset(val_df, 'val')


    def _augment_subset(self, subset_df: pd.DataFrame, dataset_type: str) -> None:
        """
        Augments a specific subset of the dataset.

        Args:
        -----
        subset_df : pd.DataFrame
            DataFrame for the dataset subset.
        dataset_type : str
            The type of dataset ('train' or 'val').
        """
        # Parse 'LesionLabel' column to lists, keeping duplicates
        def parse_labels(label_str):
            if pd.isna(label_str) or label_str == 'nolesion':
                return []
            else:
                return label_str.split(',')

        subset_df['LesionLabelList'] = subset_df['LesionLabel'].apply(parse_labels)

        # Load the augmentation counts for the current dataset type
        label_aug_counts = self.label_aug_counts[dataset_type]

        # Create a DataFrame mapping image indices to labels
        image_labels = subset_df[['LesionLabelList']].copy()

        # Prepare label queues
        label_queues = {label: [] for label in label_aug_counts.keys()}

        # Build queues of images for each label
        for idx, labels in image_labels['LesionLabelList'].items():
            for label in labels:
                if label in label_queues:
                    label_queues[label].append(idx)

        augmented_images = []

        # Augment images per label
        for label, aug_count in tqdm(label_aug_counts.items(),
                                     desc=f"Augmenting lesion images for {dataset_type}",
                                     colour='green'):
            if aug_count <= 0:
                continue
            images_with_label = label_queues[label]
            if not images_with_label:
                print(f"No images found with label {label} to augment.")
                continue
            for _ in range(aug_count):
                idx = random.choice(images_with_label)
                row = subset_df.iloc[idx]
                augmented_image = self._apply_augmentation(row, dataset_type)
                augmented_images.append(augmented_image)

        # Augment no-lesion images
        nolesion_images = subset_df[subset_df['LesionLabel'] == 'nolesion']
        num_nolesion_augment = self.augmented_nolesion_images[dataset_type]
        for _ in tqdm(range(num_nolesion_augment),
                      desc=f'Augmenting no-lesion images for {dataset_type} ...',
                      colour='green'):
            row = nolesion_images.sample().iloc[0]
            augmented_image = self._apply_augmentation(row, dataset_type)
            augmented_images.append(augmented_image)

        # Save augmented data
        augmented_df = pd.DataFrame(augmented_images, columns=subset_df.columns)
        augmented_df.to_csv(os.path.join(self.base_output_path, f'augmented_{dataset_type}.csv'), index=False)
        full_df = pd.concat([subset_df, augmented_df], ignore_index=True)
        full_df.to_csv(os.path.join(self.base_output_path, f'processed_{dataset_type}.csv'), index=False)



    def _apply_augmentation(self, row: pd.Series, dataset_type: str) -> pd.Series:
        """
        Applies a random augmentation to an image and updates the bounding box if applicable.

        Args:
        -----
        row : pd.Series
            A row from the dataset containing metadata about the image.
        dataset_type : str
            Specifies the dataset type (e.g., 'train' or 'val').

        Returns:
        --------
        pd.Series
            A new row with the updated path of the augmented image and bounding box information if applicable.
        """
        img_path = row['Frame_path']
        img = cv2.imread(img_path)
        aug_type = random.choices(list(self.augmentation_functions.keys()), self.augmentation_weights, k=1)[0]
        augmentation_func, modifies_bbox = self.augmentation_functions[aug_type]
        groundtruth_path = row['Groundtruth_path']
        lesion_label = row['LesionLabel']
        augmentation_type = aug_type

        if modifies_bbox and groundtruth_path != 'nolesion':
            img, bbox_coords = augmentation_func(img, groundtruth_path)
        else:
            img = augmentation_func(img)

        new_img_path, new_img_name = self._generate_new_path(img_path, dataset_type, augmentation_type, lesion_label)
        cv2.imwrite(new_img_path, img)

        new_gt_path = 'nolesion'
        if groundtruth_path != 'nolesion':
            if modifies_bbox:
                new_gt_path = self._save_transformed_bounding_box(bbox_coords, new_img_name, dataset_type)
            else:
                new_gt_path = self._copy_bounding_box(groundtruth_path, new_img_name, dataset_type)

        new_row = row.copy()
        new_row['Frame_path'] = new_img_path
        new_row['Groundtruth_path'] = new_gt_path
        return new_row

    # Augmentation Techniques

    def _random_brightness(self, img: np.ndarray) -> np.ndarray:
        """
        Randomly adjusts the brightness of an image.

        Args:
        -----
        img : np.ndarray
            The input image in BGR format.

        Returns:
        --------
        np.ndarray
            The brightness-adjusted image.
        """
        value = random.uniform(0.75, 1.2)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], value)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _random_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Randomly adjusts the contrast of an image.

        Args:
        -----
        img : np.ndarray
            The input image in BGR format.

        Returns:
        --------
        np.ndarray
            The contrast-adjusted image.
        """
        alpha = random.uniform(0.75, 1.20)
        return cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, 0)

    def _random_translation(self, img: np.ndarray, gt_path: str = None) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
        """
        Applies a random translation to an image, updating the bounding box if applicable.

        Args:
        -----
        img : np.ndarray
            The input image in BGR format.
        gt_path : str, optional
            Path to the ground truth bounding box file.

        Returns:
        --------
        Union[np.ndarray, Tuple[np.ndarray, List[int]]]
            The translated image, and the new bounding box coordinates if a bounding box is provided.
        """
        tx, ty = random.randint(-25, 25), random.randint(-25, 25)
        top, bottom = max(0, ty), max(0, -ty)
        left, right = max(0, tx), max(0, -tx)
        image_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image_with_border = cv2.warpAffine(image_with_border, translation_matrix,
                                                      (img.shape[1] + left + right, img.shape[0] + top + bottom))
        translated_image = translated_image_with_border[top:top + img.shape[0], left:left + img.shape[1]]

        if gt_path and gt_path != 'nolesion':
            with open(gt_path, 'r') as f:
                bbox = f.readline().strip().split()
                x, y, w, h, label = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), str(bbox[4])
            new_x, new_y = x + tx, y + ty
            return translated_image, [new_x, new_y, w, h, label]
        else:
            return translated_image

    def _xray_noise(self, img: np.ndarray) -> np.ndarray:
        """
        Adds Gaussian noise to simulate x-ray scanner noise.

        Args:
        -----
        img : np.ndarray
            The input image in BGR format.

        Returns:
        --------
        np.ndarray
            The image with added Gaussian noise.
        """
        mean, sigma = 0, 10
        gauss = np.random.normal(mean, sigma, img.shape).astype(np.uint8)
        return np.clip(img + gauss, 0, 255)

    def _elastic_deformation(self, img: np.ndarray, gt_path: str = None) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
        """
        Applies elastic deformation to the image and updates the bounding box accordingly.

        Args:
        -----
        img : np.ndarray
            The input image in BGR format.
        gt_path : str
            Path to the ground truth bounding box file.

        Returns:
        --------
        Union[np.ndarray, Tuple[np.ndarray, List[int]]]
            The deformed image and the new bounding box coordinates if applicable.

        References:
        --------
            SIMARD, Patrice Y., et al. Best practices for convolutional neural networks applied to visual document analysis. 
            En Icdar. 2003.

            Function partially inspired in https://www.kaggle.com/code/bguberfain/elastic-transform-for-data-augmentation 
        """
        image_width = img.shape[1]

        # Controls the intensity of the deformation. Higher alpha values lead to more significant warping. 
        # Typically, alpha is a multiple of the image dimensions to scale the intensity appropriately.
        alpha = np.random.uniform(0.5 * image_width, 3 * image_width)

        # Defines the smoothness of the deformation. Higher sigma values produce smoother and more gradual 
        # deformations, while lower sigma values lead to sharper and more abrupt deformations.
        sigma = np.random.uniform(0.05 * image_width, 0.2 * image_width)

        alpha_affine = img.shape[1] * 0.08

        random_state = np.random.RandomState(None)

        # Generate displacement fields
        shape = img.shape[:2]
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

        # Create meshgrid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        # Apply elastic deformation to the image
        deformed_image = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        if gt_path and gt_path != 'nolesion':
            # Read bounding box
            with open(gt_path, 'r') as f:
                bbox = f.readline().strip().split()
                x0, y0, w, h, label = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), str(bbox[4])

            # Original bounding box corners
            corners = np.array([
                [x0, y0],
                [x0 + w, y0],
                [x0, y0 + h],
                [x0 + w, y0 + h]
            ], dtype=np.float32)

            # Apply displacement to bounding box corners
            x_coords = corners[:, 0]
            y_coords = corners[:, 1]
            x_coords_deformed = x_coords + dx[y_coords.astype(int), x_coords.astype(int)]
            y_coords_deformed = y_coords + dy[y_coords.astype(int), x_coords.astype(int)]

            # New bounding box
            x_min = max(0, int(np.min(x_coords_deformed)))
            x_max = min(shape[1], int(np.max(x_coords_deformed)))
            y_min = max(0, int(np.min(y_coords_deformed)))
            y_max = min(shape[0], int(np.max(y_coords_deformed)))
            w_new = x_max - x_min
            h_new = y_max - y_min

            return deformed_image, [x_min, y_min, w_new, h_new, label]
        else:
            return deformed_image

    # Bounding Box Handling

    def _save_transformed_bounding_box(self, bbox_coords: List[int], new_img_name: str, dataset_type: str) -> str:
        """
        Saves updated bounding box coordinates after augmentation.

        Args:
        -----
        bbox_coords : List[int]
            Coordinates of the transformed bounding box [x, y, width, height, label].
        new_img_name : str
            Name of the augmented image file.
        dataset_type : str
            Dataset type ('train' or 'val').

        Returns:
        --------
        str
            Path to the new bounding box file.
        """
        path_parts = new_img_name.split('_')
        patient_video = path_parts[0] + '_' + path_parts[1]
        new_dir = os.path.join(self.base_output_path, dataset_type, 'labels', 'lesion', patient_video)
        os.makedirs(new_dir, exist_ok=True)
        gt_name = new_img_name.replace('.png', '.txt')
        new_gt_path = os.path.join(new_dir, gt_name)
        with open(new_gt_path, 'w') as f:
            f.write(f"{bbox_coords[0]} {bbox_coords[1]} {bbox_coords[2]} {bbox_coords[3]} {bbox_coords[4]}")
        return new_gt_path

    def _copy_bounding_box(self, gt_path: str, new_img_name: str, dataset_type: str) -> str:
        """
        Copies the existing bounding box file to the new augmented image's location.

        Args:
        -----
        gt_path : str
            Original path of the bounding box file.
        new_img_name : str
            Name of the augmented image file.
        dataset_type : str
            Dataset type ('train' or 'val').

        Returns:
        --------
        str
            Path to the copied bounding box file.
        """
        path_parts = new_img_name.split('_')
        patient_video = path_parts[0] + '_' + path_parts[1]
        new_dir = os.path.join(self.base_output_path, dataset_type, 'labels', 'lesion', patient_video)
        os.makedirs(new_dir, exist_ok=True)
        gt_name = new_img_name.replace('.png', '.txt')
        new_gt_path = os.path.join(new_dir, gt_name)
        shutil.copy(gt_path, new_gt_path)
        return new_gt_path

    # Path Generation

    def _generate_new_path(self, img_path: str, dataset_type: str, augmentation_type: str, lesion_label: str) -> Tuple[str, str]:
        """
        Generates a unique file path for the augmented image.

        Args:
        -----
        img_path : str
            Original path of the image to be augmented.
        dataset_type : str
            Specifies the dataset type ('train' or 'val').
        augmentation_type : str
            The type of augmentation applied.
        lesion_label : str
            The lesion label associated with the image.

        Returns:
        --------
        Tuple[str, str]
            The new path for the augmented image and the new filename.
        """
        path_parts = img_path.split('/')
        patient_video = path_parts[-4] + "_" + path_parts[-3]
        sub_dir = 'lesion' if lesion_label != 'nolesion' else 'nolesion'
        base_dir = os.path.join(self.base_output_path, dataset_type, 'images', sub_dir, patient_video)
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

    Args:
    -----
    train_path : str
        Path to the training dataset CSV.
    val_path : str
        Path to the validation dataset CSV.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        DataFrames for the training and validation datasets.
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    return train_df, val_df


def run_augmentData(train_df: pd.DataFrame, val_df: pd.DataFrame, base_output_path: str,
                    augmented_lesion_images: int, augmented_nolesion_images: int, ignore_top_n: int = 0, seed: int = 42) -> None:
    """
    Augments training and validation datasets by applying various augmentations.

    Args:
    -----
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
    seed : int, optional
        Random seed for reproducibility.
    """
    augmentor = Augmentor(base_output_path, augmented_lesion_images, augmented_nolesion_images, ignore_top_n, seed)
    augmentor.augment_data(train_df, val_df)
