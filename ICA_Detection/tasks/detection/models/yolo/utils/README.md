# NPZ Loader Experiment Documentation

## Overview

This module provides utilities for working with mask-guided attention in YOLO object detection models. It enables the integration of image masks with standard images by:

1. Loading and processing `.npz` files containing paired image-mask data
2. Converting traditional YOLO datasets to mask-guided attention datasets
3. Verifying dataset integrity

The primary use case is enhancing YOLO models to focus on specific regions of interest using accompanying segmentation masks.

## Installation Requirements

```
numpy
Pillow
torch
tqdm
```

## Module Components

### 1. `load_npz_data(npz_path)`

Extracts image and mask arrays from a `.npz` file.

**Parameters:**
- `npz_path` (str or Path): Path to the `.npz` file

**Returns:**
- `tuple`: (image, mask) where both are numpy arrays

**Notes:**
- Supports standard key names ('image', 'mask') or default numpy keys ('arr_0', 'arr_1')
- Automatically expands 2D masks to 3D with a channel dimension

### 2. `preprocess_npz_for_yolo(npz_path, imgsz=640)`

Prepares a `.npz` file for YOLO model inference by loading, normalizing, and formatting.

**Parameters:**
- `npz_path` (str or Path): Path to the `.npz` file
- `imgsz` (int): Target image size (reserved for future implementation)

**Returns:**
- `torch.Tensor`: Preprocessed tensor with combined image and mask channels

**Processing Steps:**
1. Normalizes image values to [0, 1]
2. Normalizes mask values to [0, 1]
3. Converts image from HWC to CHW format if needed
4. Concatenates image and mask along channel dimension
5. Returns a batched tensor ready for model input

### 3. `create_mask_attention_dataset(...)`

Creates a new YOLO dataset with image-mask pairs in NPZ format by mapping between datasets.

**Parameters:**
- `images_dataset_root`: Path to YOLO dataset with images
- `labels_dataset_root`: Path to YOLO dataset with labels
- `mask_folder`: Path to folder containing mask images
- `mapping_json`: Path to JSON mapping between label images and dataset images
- `output_root`: Name/path for the output dataset folder
- `splits`: Dataset splits to process (default: train, val, test)
- `mask_suffix`: Suffix for mask filenames (default: "_seg")
- `mask_extension`: File extension for mask files (default: ".png")

**Returns:**
- `dict`: Statistics about the conversion process

**Key Features:**
- Creates a complete YOLO-compatible dataset structure
- Uses JSON mapping to connect labels with correct image-mask pairs
- Preserves dataset split organization
- Provides detailed statistics about the conversion

### 4. `verify_npz_dataset(dataset_root, splits=("train", "val", "test"))`

Validates that all NPZ files in a dataset contain proper image and mask data.

**Parameters:**
- `dataset_root` (str or Path): Root directory of the YOLO dataset
- `splits` (tuple): Dataset splits to verify

**Returns:**
- `bool`: True if all files are valid, False otherwise

**Checks:**
- Verifies each NPZ file can be loaded
- Confirms both image and mask data are present
- Reports any invalid or corrupted files

## Dataset Format

### Input Requirements

1. **Image Dataset**: Standard YOLO folder structure with images in `images/{split}/`
2. **Label Dataset**: Standard YOLO folder structure with labels in `labels/{split}/`
3. **Mask Folder**: Directory with mask images, named with suffix (e.g., `image_name_seg.png`)
4. **JSON Mapping**: File with format `{"label_image.png": "dataset_image.png", ...}`

### Output Format

1. **Images**: NPZ files containing both image and mask arrays (`image` and `mask` keys)
2. **Labels**: Standard YOLO format text files with bounding box annotations
3. **Structure**: Standard YOLO folder structure preserved:
   ```
   mask_attention_yolo/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── labels/
       ├── train/
       ├── val/
       └── test/
   ```

## Example Usage

```python
# Basic usage for loading a single NPZ file
image, mask = load_npz_data("/path/to/data.npz")

# Preprocess a single NPZ file for YOLO inference
tensor = preprocess_npz_for_yolo("/path/to/data.npz")

# Convert a dataset to mask-attention format
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
verify_npz_dataset("mask_attention_yolo")
```

## Advanced Usage

### Example: Converting a Multi-Split Dataset with Custom Configuration

```python
import json
from pathlib import Path

# Create custom mapping
mapping = {}
for img_path in Path("dataset/images/train").glob("*.jpg"):
    # Map each image to itself (simplified example)
    mapping[img_path.name] = img_path.name

# Save mapping
with open("mapping.json", "w") as f:
    json.dump(mapping, f)

# Create mask-attention dataset with custom settings
create_mask_attention_dataset(
    images_dataset_root="dataset",
    labels_dataset_root="another_dataset",
    mask_folder="segmentation_masks",
    mapping_json="mapping.json",
    output_root="mask_guided_dataset",
    splits=["train", "val"],  # Skip test split
    mask_suffix="_segmentation",
    mask_extension=".png"
)
```

## Troubleshooting

### Common Issues

1. **Missing Masks**: Ensure mask files follow the naming convention (`{image_stem}{mask_suffix}{mask_extension}`)
2. **JSON Format**: Verify the JSON mapping has the format `{"label_image.png": "dataset_image.png", ...}`
3. **Dataset Structure**: Confirm both input datasets follow standard YOLO structure
4. **Image Formats**: Check that images are readable by PIL (jpg, png, bmp, etc.)
5. **Path Issues**: Use absolute paths or correct relative paths for all directories

### Statistics Interpretation

The `create_mask_attention_dataset` function returns statistics to help identify issues:
- `processed`: Successfully created image-mask pairs
- `labels_copied`: Labels copied to the new dataset
- `masks_not_found`: Images without corresponding masks
- `images_not_found`: Entries in the JSON mapping not found in the image dataset
- `labels_not_found`: Entries in the JSON mapping not found in the label dataset
- `errors`: Unexpected errors during processing

## Best Practices

1. **Backup Data**: Always work with copies of your original datasets
2. **Verify Results**: Run `verify_npz_dataset()` after dataset creation
3. **Check Samples**: Manually inspect a few NPZ files to confirm proper content
4. **Memory Management**: For large datasets, process in batches or splits

## License and Attribution

This module was created for use with the Mask-Guided YOLO project for coronary angiography detection.