import numpy as np
from PIL import Image
import torch
from pathlib import Path
import os
import shutil
from tqdm import tqdm

def load_npz_data(npz_path):
    """
    Load image and mask from a .npz file.
    
    Args:
        npz_path (str): Path to the .npz file
        
    Returns:
        tuple: (image, mask) where both are numpy arrays
    """
    data = np.load(npz_path)
    image = data['image'] if 'image' in data else data['arr_0']
    mask = data['mask'] if 'mask' in data else data['arr_1']
    
    # Ensure mask has proper dimensions
    if mask.ndim == 2:
        mask = mask[..., np.newaxis]
    
    return image, mask

def preprocess_npz_for_yolo(npz_path, imgsz=640):
    """
    Load and preprocess .npz file for YOLOv8 inference.
    
    Args:
        npz_path (str): Path to the .npz file
        imgsz (int): Target image size
        
    Returns:
        torch.Tensor: Preprocessed tensor with image and mask channels
    """
    image, mask = load_npz_data(npz_path)
    
    # Normalize image to [0, 1]
    if image.max() > 1.0:
        image = image / 255.0
        
    # Normalize mask to [0, 1]
    if mask.max() > 1.0:
        mask = mask / 255.0
    elif mask.max() > 0:  # Ensure mask is normalized if not binary
        mask = mask / mask.max()
    
    # Convert to torch tensor and add batch dimension
    if image.shape[-1] == 3:  # If image is HWC format
        image = image.transpose(2, 0, 1)  # Convert to CHW format
        
    # Add mask as additional channel
    if mask.ndim == 3 and mask.shape[0] == 1:
        # Mask is already in CHW format
        combined = np.concatenate((image, mask), axis=0)
    else:
        # Convert mask to CHW format
        mask = mask.transpose(2, 0, 1) if mask.ndim == 3 else mask[np.newaxis, ...]
        combined = np.concatenate((image, mask), axis=0)
    
    # Convert to torch tensor
    tensor = torch.from_numpy(combined).float()
    
    # Add batch dimension if it doesn't exist
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        
    return tensor

def convert_dataset_to_npz(
    dataset_root, 
    mask_folder, 
    splits=("train", "val", "test"),
    mask_suffix="_seg",
    mask_extension=".png",
    backup=True
):
    """
    Convert a YOLO-style dataset to include masks in npz format.
    
    Args:
        dataset_root (str): Root directory of the YOLO dataset (contains images/ and labels/)
        mask_folder (str): Directory containing mask files
        splits (tuple): Dataset splits to process (default: ("train", "val", "test"))
        mask_suffix (str): Suffix added to image name to find mask (default: "_seg")
        mask_extension (str): File extension for mask files (default: ".png")
        backup (bool): Whether to create backup of original images (default: True)
        
    Returns:
        dict: Statistics about the conversion process
    """
    # Initialize statistics
    stats = {"processed": 0, "skipped": 0, "errors": 0}
    
    # Create dataset directory path objects
    dataset_root = Path(dataset_root)
    mask_folder = Path(mask_folder)
    
    # Create backup folder if needed
    if backup:
        backup_folder = dataset_root / "images_backup"
        backup_folder.mkdir(exist_ok=True)
    
    # Process each split
    for split in splits:
        images_dir = dataset_root / "images" / split
        
        # Skip if directory doesn't exist
        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist, skipping.")
            continue
            
        # Create backup for this split
        if backup:
            split_backup = backup_folder / split
            split_backup.mkdir(exist_ok=True)
        
        # Process each image in the directory
        image_paths = sorted(images_dir.glob("*.*"))
        print(f"Processing {len(image_paths)} images in {split} split...")
        
        for img_path in tqdm(image_paths):
            # Skip if already an npz file
            if img_path.suffix.lower() == '.npz':
                stats["skipped"] += 1
                continue
                
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                stats["skipped"] += 1
                continue
                
            try:
                # Construct mask path
                mask_name = img_path.stem + mask_suffix + mask_extension
                mask_path = mask_folder / mask_name
                
                # Check if mask exists
                if not mask_path.exists():
                    print(f"Warning: Mask {mask_path} not found for {img_path}")
                    stats["skipped"] += 1
                    continue
                
                # Load image and mask
                image = np.array(Image.open(img_path))
                mask = np.array(Image.open(mask_path))
                
                # Create backup if needed
                if backup:
                    shutil.copy(img_path, split_backup / img_path.name)
                
                # Create npz file with the same base name
                npz_path = img_path.parent / (img_path.stem + '.npz')
                np.savez_compressed(npz_path, image=image, mask=mask)
                
                # Remove original image file
                img_path.unlink()
                
                stats["processed"] += 1
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                stats["errors"] += 1
    
    print(f"Conversion completed: {stats['processed']} processed, {stats['skipped']} skipped, {stats['errors']} errors")
    return stats

def verify_npz_dataset(dataset_root, splits=("train", "val", "test")):
    """
    Verify that all NPZ files in the dataset contain valid image and mask data.
    
    Args:
        dataset_root (str): Root directory of the YOLO dataset
        splits (tuple): Dataset splits to verify
        
    Returns:
        bool: True if all files are valid, False otherwise
    """
    dataset_root = Path(dataset_root)
    all_valid = True
    
    for split in splits:
        images_dir = dataset_root / "images" / split
        if not images_dir.exists():
            continue
            
        print(f"Verifying {split} split...")
        for npz_path in tqdm(list(images_dir.glob("*.npz"))):
            try:
                # Try to load the NPZ file
                data = np.load(npz_path)
                
                # Check if required keys exist
                if 'image' not in data and 'arr_0' not in data:
                    print(f"Error: Missing image data in {npz_path}")
                    all_valid = False
                    
                if 'mask' not in data and 'arr_1' not in data:
                    print(f"Error: Missing mask data in {npz_path}")
                    all_valid = False
                
            except Exception as e:
                print(f"Error loading {npz_path}: {e}")
                all_valid = False
                
    if all_valid:
        print("All NPZ files are valid.")
    else:
        print("Some NPZ files have issues. Check the error messages above.")
        
    return all_valid


"""
Example:

# Convert dataset
convert_dataset_to_npz(
    dataset_root="/path/to/dataset",  # Contains images/ and labels/ folders
    mask_folder="/path/to/masks",     # Contains mask images with _seg suffix
    splits=["train", "val", "test"],
    mask_suffix="_seg",
    mask_extension=".png",
    backup=True  # Creates backup of original images in images_backup/
)

# Verify the converted dataset
verify_npz_dataset("/path/to/dataset")
"""