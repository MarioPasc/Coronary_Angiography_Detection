import json
import os
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, List, Any, Tuple


def create_segmentation_dataset(
    input_json_path: str,
    output_json_path: str,
    output_masks_dir: str,
    image_size: Tuple[int, int] = (512, 512),  # Standard size for masks
) -> None:
    """
    Create a standardized segmentation dataset from the original JSON format.

    Args:
        input_json_path: Path to the original JSON file with vessel_segmentations
        output_json_path: Path to save the standardized JSON file
        output_masks_dir: Directory to save the binary masks
        image_size: Size to resize all images and masks for consistency
    """
    # Create masks directory if it doesn't exist
    os.makedirs(output_masks_dir, exist_ok=True)

    # Load the original JSON data
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # Get the actual dataset (handle both formats)
    if "Standard_dataset" in data:
        data = data["Standard_dataset"]

    # Initialize new standardized dataset
    segmentation_dataset = []

    # Process each image in the dataset
    for image_id, entry in data.items():
        # Skip entries without vessel segmentations
        annotations = entry.get("annotations", {})
        vessel_segmentations = annotations.get("vessel_segmentations", [])

        if not vessel_segmentations:
            continue

        # Get image information
        image_info = entry.get("image", {})
        image_path = (
            image_info.get("route")
            or image_info.get("dataset_route")
            or image_info.get("original_route")
        )

        if not image_path or not os.path.exists(image_path):
            print(f"Image file not found at {image_path}")
            continue

        # Get original image dimensions
        original_width = image_info.get("width")
        original_height = image_info.get("height")

        if not original_width or not original_height:
            # Try to get dimensions from the image file
            try:
                with Image.open(image_path) as img:
                    original_width, original_height = img.size
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                continue

        mask_filename = f"{image_id}_mask.png"
        mask_path = os.path.join(output_masks_dir, mask_filename)

        # Add to segmentation dataset
        segmentation_entry = {
            "id": image_id,
            "image_path": image_path,
            "mask_path": mask_path,
            "original_size": (original_width, original_height),
            "processed_size": image_size,
        }

        segmentation_dataset.append(segmentation_entry)

    # Save standardized JSON
    with open(output_json_path, "w") as f:
        json.dump(
            {
                "dataset_info": {
                    "description": "Vessel segmentation dataset",
                    "task": "binary_segmentation",
                    "total_samples": len(segmentation_dataset),
                },
                "samples": segmentation_dataset,
            },
            f,
            indent=2,
        )

    print(
        f"Created standardized segmentation dataset with {len(segmentation_dataset)} samples"
    )
    print(f"JSON saved to: {output_json_path}")
    print(f"Masks saved to: {output_masks_dir}")


create_segmentation_dataset(
    input_json_path="/home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/json_metadata/processed.json",
    output_json_path="/home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/datasets/seg/segmentation_ds.json",
    output_masks_dir="/home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/datasets/seg",
)
