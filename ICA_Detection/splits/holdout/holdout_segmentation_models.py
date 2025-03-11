import os
import json
import re
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

DEBUG = False


class VesselSegmentationDataset(Dataset):
    """
    Dataset for vessel segmentation using mask images.
    This class loads images and their corresponding masks for segmentation tasks.
    """

    def __init__(self, json_data_path, image_root, masks_root, transforms=None):
        """
        Args:
            json_data_path (str): Path to the JSON file containing dataset information
            image_root (str): Directory containing the images
            masks_root (str): Directory containing the segmentation masks
            transforms (callable, optional): Optional transforms to be applied on a sample
        """
        super().__init__()
        self.image_root = image_root
        self.masks_root = masks_root
        self.transforms = transforms

        # Load the dataset JSON
        with open(json_data_path, "r") as f:
            self.data = json.load(f)

        # Extract all image IDs with segmentation data
        self.image_ids = []
        for dataset_key, dataset_data in self.data.items():
            for image_id, image_data in dataset_data.items():
                # Check if the image has vessel segmentations
                if (
                    "annotations" in image_data
                    and "vessel_segmentations" in image_data["annotations"]
                    and len(image_data["annotations"]["vessel_segmentations"]) > 0
                ):
                    self.image_ids.append((dataset_key, image_id))

        if DEBUG:
            print(
                f"[DEBUG] Loaded {len(self.image_ids)} images with vessel segmentations"
            )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Get image ID and dataset info
        dataset_key, image_id = self.image_ids[idx]
        img_data = self.data[dataset_key][image_id]

        # Get image info
        img_info = img_data["image"]
        img_path = os.path.join(self.image_root, img_info["name"])

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Get mask path from the first vessel segmentation
        # (All segmentations point to the same mask file)
        mask_path = None
        if len(img_data["annotations"]["vessel_segmentations"]) > 0:
            mask_path = img_data["annotations"]["vessel_segmentations"][0][
                "attributes"
            ]["mask_path"]
            # Extract just the filename from the mask path
            mask_filename = os.path.basename(mask_path)
            # Create the new path in the masks_root directory
            mask_path = os.path.join(self.masks_root, mask_filename)

        # Load mask
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        else:
            # Create an empty mask if no mask file exists
            width, height = img.size
            mask = Image.new("L", (width, height), 0)
            if DEBUG:
                print(f"[DEBUG] No mask found for {image_id}, creating empty mask")

        # Apply transforms if available
        if self.transforms:
            img, mask = self.transforms(img, mask)
        else:
            # Default transforms
            img = transforms.ToTensor()(img)
            mask = transforms.ToTensor()(mask)

            # Normalize mask to binary (0 or 1)
            mask = (mask > 0.5).float()

        return img, mask


class SegmentationTransforms:
    """
    Transforms for segmentation data that apply the same transformations
    to both image and mask.
    """

    def __init__(self, img_size=512, augment=False):
        self.img_size = img_size
        self.augment = augment

    def __call__(self, img, mask):
        # Convert PIL images to numpy arrays
        img_np = np.array(img)
        mask_np = np.array(mask)

        # Resize
        if img_np.shape[0] != self.img_size or img_np.shape[1] != self.img_size:
            img_np = cv2.resize(
                img_np, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
            )
            mask_np = cv2.resize(
                mask_np, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST
            )

        # Data augmentation
        if self.augment:
            # Randomly flip horizontally
            if np.random.rand() > 0.5:
                img_np = np.fliplr(img_np)
                mask_np = np.fliplr(mask_np)

            # Randomly flip vertically
            if np.random.rand() > 0.5:
                img_np = np.flipud(img_np)
                mask_np = np.flipud(mask_np)

            # Random brightness/contrast
            if np.random.rand() > 0.5:
                brightness = 0.7 + np.random.rand() * 0.6  # 0.7-1.3
                img_np = np.clip(img_np * brightness, 0, 255).astype(np.uint8)

        # Default transforms
        img_tensor = transforms.ToTensor()(
            np.array(img)
        )  # Convert PIL to numpy first if needed
        mask_tensor = transforms.ToTensor()(
            np.array(mask).copy()
        )  # Add .copy() to prevent stride issues
        # Normalize image
        img_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
            img_tensor
        )

        # Ensure mask is binary
        mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor


def holdout_segmentation(
    json_data_path,
    splits_info_path,
    images_root,
    masks_root,
    img_size=512,
    augment_train=True,
    batch_size=4,
    shuffle_train=True,
    num_workers=4,
):
    """
    Load a dataset, separate it into train/val/test subsets by patient key,
    and return DataLoaders for segmentation.

    Args:
        json_data_path (str): Path to the JSON file with dataset info
        splits_info_path (str): Path to the JSON file with train/val/test splits
        images_root (str): Directory containing the images
        masks_root (str): Directory containing segmentation masks
        img_size (int): Size to resize images to
        augment_train (bool): Whether to apply data augmentation to the training set
        batch_size (int): Batch size for DataLoaders
        shuffle_train (bool): Whether to shuffle the training data
        num_workers (int): Number of workers for data loading

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1) Read splits info
    if not os.path.isfile(splits_info_path):
        raise FileNotFoundError(f"Splits info JSON not found: {splits_info_path}")

    with open(splits_info_path, "r") as f:
        splits_info = json.load(f)

    # Build a dict: "patient_key" -> "train"/"val"/"test"
    patient_to_split = {}
    for split_name, dataset_map in splits_info.items():
        for _, patient_list in dataset_map.items():
            for pkey in patient_list:
                patient_to_split[pkey] = split_name

    # 2) Load the dataset info
    with open(json_data_path, "r") as f:
        dataset_info = json.load(f)

    # 3) Split the data into train, val, test sets
    train_ids = []
    val_ids = []
    test_ids = []

    for dataset_key, dataset_data in dataset_info.items():
        for image_id, image_data in dataset_data.items():
            # Check if this image has vessel segmentations
            if (
                "annotations" not in image_data
                or "vessel_segmentations" not in image_data["annotations"]
                or len(image_data["annotations"]["vessel_segmentations"]) == 0
            ):
                continue

            # Parse the patient key from the image ID (e.g., "arcadetrain_p676_v676_00676" -> "arcadetrain_p676")
            match = re.match(r"([a-zA-Z0-9]+_[a-zA-Z0-9]+)_.*", image_id)
            if match:
                pkey = match.group(1)  # e.g. "arcadetrain_p676"
            else:
                # Fallback if we can't parse
                pkey = "unknown_unknown"

            # Determine which split it belongs to
            split_name = patient_to_split.get(pkey, None)
            if split_name == "train":
                train_ids.append((dataset_key, image_id))
            elif split_name == "val":
                val_ids.append((dataset_key, image_id))
            elif split_name == "test":
                test_ids.append((dataset_key, image_id))

    if DEBUG:
        print(
            f"[DEBUG] Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}"
        )

    # 4) Create custom datasets for each split
    class SplitSegmentationDataset(Dataset):
        def __init__(
            self, json_data_path, image_root, masks_root, image_ids, transforms=None
        ):
            super().__init__()
            self.image_root = image_root
            self.masks_root = masks_root
            self.transforms = transforms

            # Load dataset info
            with open(json_data_path, "r") as f:
                self.data = json.load(f)

            self.image_ids = image_ids

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, idx):
            # Get image ID and dataset info
            dataset_key, image_id = self.image_ids[idx]
            img_data = self.data[dataset_key][image_id]

            # Get image info
            img_info = img_data["image"]
            img_path = os.path.join(self.image_root, img_info["name"])

            # Load image
            img = Image.open(img_path).convert("RGB")

            # Get mask path from the first vessel segmentation
            mask_path = None
            if len(img_data["annotations"]["vessel_segmentations"]) > 0:
                mask_path = img_data["annotations"]["vessel_segmentations"][0][
                    "attributes"
                ]["mask_path"]
                # Extract just the filename from the mask path
                mask_filename = os.path.basename(mask_path)
                # Create the new path in the masks_root directory
                mask_path = os.path.join(self.masks_root, mask_filename)

            # Load mask
            if mask_path and os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")  # Convert to grayscale
            else:
                # Create an empty mask if no mask file exists
                width, height = img.size
                mask = Image.new("L", (width, height), 0)
                if DEBUG:
                    print(f"[DEBUG] No mask found for {image_id}, creating empty mask")

            # Apply transforms if available
            if self.transforms:
                img, mask = self.transforms(img, mask)
            else:
                # Default transforms
                # Default transforms
                img = transforms.ToTensor()(
                    np.array(img)
                )  # Convert PIL to numpy first if needed
                mask = transforms.ToTensor()(
                    np.array(mask).copy()
                )  # Add .copy() to prevent stride issues

                # Normalize mask to binary (0 or 1)
                mask = (mask > 0.5).float()

            return img, mask

    # 5) Create transforms
    train_transforms = SegmentationTransforms(img_size=img_size, augment=augment_train)
    val_transforms = SegmentationTransforms(img_size=img_size, augment=False)

    # 6) Create datasets
    train_dataset = SplitSegmentationDataset(
        json_data_path=json_data_path,
        image_root=images_root,
        masks_root=masks_root,
        image_ids=train_ids,
        transforms=train_transforms,
    )

    val_dataset = SplitSegmentationDataset(
        json_data_path=json_data_path,
        image_root=images_root,
        masks_root=masks_root,
        image_ids=val_ids,
        transforms=val_transforms,
    )

    test_dataset = None
    if len(test_ids) > 0:
        test_dataset = SplitSegmentationDataset(
            json_data_path=json_data_path,
            image_root=images_root,
            masks_root=masks_root,
            image_ids=test_ids,
            transforms=val_transforms,
        )

    # 7) Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    return train_loader, val_loader, test_loader


def visualize_sample(image, mask, title=None):
    """Helper function to visualize an image and its mask"""
    # Convert tensor to numpy
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        # Denormalize if needed
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().cpu().numpy()

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Mask")
    ax[1].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Paths
    json_data_path = "path/to/dataset.json"
    splits_info_path = "path/to/splits_info.json"
    images_root = "path/to/images"
    masks_root = "path/to/masks"

    # Get dataloaders
    train_loader, val_loader, test_loader = holdout_segmentation(
        json_data_path=json_data_path,
        splits_info_path=splits_info_path,
        images_root=images_root,
        masks_root=masks_root,
        batch_size=4,
    )

    # Print dataset sizes
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        print(f"Test samples: {len(test_loader.dataset)}")

    # Visualize a sample
    imgs, masks = next(iter(train_loader))
    visualize_sample(imgs[0], masks[0], title="Sample from Training Set")
