

import numpy as np
from PIL import Image
import torch
from pathlib import Path
import json
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

def create_mask_attention_dataset(
    images_dataset_root,  # YOLO dataset with images but no labels
    labels_dataset_root,  # YOLO dataset with labels
    mask_folder,          # Folder containing mask images
    mapping_json,         # JSON mapping: label_image -> image
    output_root="mask_attention_yolo",  # Output dataset name
    splits=("train", "val", "test"),
    mask_suffix="_seg",
    mask_extension=".png"
):
    """
    Create a new YOLO dataset with image-mask pairs in NPZ format and corresponding labels.
    
    Args:
        images_dataset_root (str): Root directory of YOLO dataset with images
        labels_dataset_root (str): Root directory of YOLO dataset with labels
        mask_folder (str): Directory containing mask files
        mapping_json (str): Path to JSON file mapping label images to actual images
        output_root (str): Name of the output dataset folder
        splits (tuple): Dataset splits to process (default: ("train", "val", "test"))
        mask_suffix (str): Suffix added to image name to find mask (default: "_seg")
        mask_extension (str): File extension for mask files (default: ".png")
        
    Returns:
        dict: Statistics about the conversion process
    """
    # Initialize statistics
    stats = {"processed": 0, "labels_copied": 0, "masks_not_found": 0, 
             "images_not_found": 0, "labels_not_found": 0, "errors": 0}
    
    # Create path objects
    images_dataset_root = Path(images_dataset_root)
    labels_dataset_root = Path(labels_dataset_root)
    mask_folder = Path(mask_folder)
    output_root = Path(output_root)
    
    # Create output directory structure
    for split in splits:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Load JSON mapping
    with open(mapping_json, 'r') as f:
        mapping = json.load(f)
    
    print(f"Processing {len(mapping)} entries from mapping file...")
    
    # Process each mapping entry
    for label_image, image_name in tqdm(mapping.items()):
        try:
            # Determine which split this image belongs to
            split_found = None
            img_path = None
            
            # Search for the image in all splits
            for split in splits:
                test_path = images_dataset_root / "images" / split / image_name
                if test_path.exists():
                    split_found = split
                    img_path = test_path
                    break
            
            if split_found is None:
                print(f"Warning: Image {image_name} not found in any split")
                stats["images_not_found"] += 1
                continue
            
            # Construct mask path
            mask_name = img_path.stem + mask_suffix + mask_extension
            mask_path = mask_folder / mask_name
            
            # Check if mask exists
            if not mask_path.exists():
                print(f"Warning: Mask {mask_path} not found for {img_path}")
                stats["masks_not_found"] += 1
                continue
            
            # Load image and mask
            image = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))
            
            # Create npz file in output directory
            output_npz = output_root / "images" / split_found / (img_path.stem + '.npz')
            np.savez_compressed(output_npz, image=image, mask=mask)
            
            # Find and copy the label file
            found_label = False
            
            for label_split in splits:
                label_path = labels_dataset_root / "labels" / label_split / (Path(label_image).stem + '.txt')
                if label_path.exists():
                    # Copy and rename the label file
                    target_label_path = output_root / "labels" / split_found / (img_path.stem + '.txt')
                    shutil.copy(label_path, target_label_path)
                    found_label = True
                    stats["labels_copied"] += 1
                    break
            
            if not found_label:
                print(f"Warning: Label for {label_image} not found")
                stats["labels_not_found"] += 1
                continue
                
            stats["processed"] += 1
                
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            stats["errors"] += 1
    
    print(f"Dataset creation completed:")
    print(f"- {stats['processed']} image-mask pairs processed")
    print(f"- {stats['labels_copied']} labels copied")
    print(f"- {stats['images_not_found']} images not found")
    print(f"- {stats['masks_not_found']} masks not found")
    print(f"- {stats['labels_not_found']} labels not found")
    print(f"- {stats['errors']} errors occurred")
    
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
create_mask_attention_dataset(
    images_dataset_root="/path/to/first/dataset",   # YOLO dataset with images
    labels_dataset_root="/path/to/second/dataset",  # YOLO dataset with labels
    mask_folder="/path/to/masks",                   # Contains mask images with _seg suffix
    mapping_json="/path/to/mapping.json",           # JSON mapping labels to images
    output_root="mask_attention_yolo",              # Output dataset name
    splits=["train", "val", "test"],
    mask_suffix="_seg",
    mask_extension=".png"
)

# Verify the converted dataset
verify_npz_dataset("/path/to/dataset")
"""