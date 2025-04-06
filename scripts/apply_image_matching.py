#!/usr/bin/env python3

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_mapping_dictionary(dict_path: str) -> Dict[str, str]:
    """
    Load the image mapping dictionary from a JSON file.

    Args:
        dict_path: Path to the JSON file containing the mapping dictionary

    Returns:
        Dictionary mapping stenosis image names to vessel segmentation mask names
    """
    with open(dict_path, "r") as f:
        data = json.load(f)
    return data.get("matches", {})


def verify_files_exist(
    mapping_dict: Dict[str, str], vessel_mask_folder: str
) -> Tuple[List[str], List[str]]:
    """
    Verify that all mask files referenced in the mapping dictionary exist.

    Args:
        mapping_dict: Dictionary mapping stenosis image names to vessel segmentation mask names
        vessel_mask_folder: Path to the folder containing vessel segmentation masks

    Returns:
        Tuple containing lists of found and missing mask files
    """
    found_masks = []
    missing_masks = []

    for stenosis_name, mask_name in mapping_dict.items():
        # Add _seg.png suffix to mask name
        mask_base = os.path.splitext(mask_name)[0]
        mask_with_suffix = f"{mask_base}_seg.png"
        mask_path = os.path.join(vessel_mask_folder, mask_with_suffix)

        if os.path.isfile(mask_path):
            found_masks.append(mask_name)
        else:
            missing_masks.append(mask_name)

    return found_masks, missing_masks


def copy_and_rename_masks(
    output_folder: str, vessel_mask_folder: str, mapping_dict: Dict[str, str]
) -> Tuple[int, int]:
    """
    Copy vessel segmentation masks to an output folder and rename them to match stenosis images.

    Args:
        output_folder: Path to the folder where renamed masks will be stored
        vessel_mask_folder: Path to the folder containing vessel segmentation masks
        mapping_dict: Dictionary mapping stenosis image names to vessel segmentation mask names

    Returns:
        Tuple containing count of successful and failed operations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Keep track of successful and failed operations
    successful = 0
    failed = 0

    # Process each entry in the mapping dictionary
    for stenosis_name, mask_name in mapping_dict.items():
        # Add _seg.png suffix to mask name
        mask_base = os.path.splitext(mask_name)[0]
        mask_with_suffix = f"{mask_base}_seg.png"
        mask_path = os.path.join(vessel_mask_folder, mask_with_suffix)
        output_path = os.path.join(output_folder, stenosis_name)

        # Check if mask file exists
        if os.path.isfile(mask_path):
            try:
                # Copy and rename the mask file
                shutil.copy2(mask_path, output_path)
                successful += 1
            except Exception as e:
                print(f"Error copying {mask_path} to {output_path}: {e}")
                failed += 1
        else:
            print(f"Mask file not found: {mask_path}")
            failed += 1

    return successful, failed


def main(
    output_folder: str, stenosis_folder: str, vessel_mask_folder: str, mapping_file: str
) -> None:
    """
    Main function to execute the mask copying and renaming process.

    Args:
        output_folder: Path to the output folder
        stenosis_folder: Path to the stenosis images folder
        vessel_mask_folder: Path to the vessel segmentation masks folder
        mapping_file: Path to the JSON file containing the mapping information

    Returns:
        None
    """
    print("Starting mask renaming process...")

    # Load the mapping dictionary
    print(f"Loading mapping dictionary from {mapping_file}")
    mapping_dict = load_mapping_dictionary(mapping_file)

    # Check if mapping dictionary is empty
    if not mapping_dict:
        print("Error: Empty mapping dictionary or invalid format")
        return

    # Print summary of the operation
    print(f"Found {len(mapping_dict)} image pairs in mapping dictionary")

    # Verify files exist
    print("Verifying mask files exist...")
    found_masks, missing_masks = verify_files_exist(mapping_dict, vessel_mask_folder)

    if missing_masks:
        print(f"Warning: {len(missing_masks)} mask files are missing")
        if len(missing_masks) <= 10:
            for mask in missing_masks:
                print(f"  - Missing mask: {mask}")
        else:
            print(f"  - First 10 missing masks: {', '.join(missing_masks[:10])}")

    # Execute the copy and rename operation
    print(f"Copying and renaming {len(found_masks)} mask files...")
    successful, failed = copy_and_rename_masks(
        output_folder, vessel_mask_folder, mapping_dict
    )

    # Print final summary
    print("\nOperation completed:")
    print(f"  - {successful} files processed successfully")
    print(f"  - {failed} operations failed")
    print(f"  - Output directory: {output_folder}")


if __name__ == "__main__":
    output = (
        "/home/mariopasc/Python/Datasets/COMBINED/tasks/stenosis_detection/labels/masks"
    )
    stenosis_path = (
        "/home/mariopasc/Python/Datasets/COMBINED/tasks/stenosis_detection/images"
    )
    vessel_path = (
        "/home/mariopasc/Python/Datasets/COMBINED/tasks/arteries_segmentation/labels"
    )
    mapping = "ICA_Detection/cfg/image_matching_ARCADE.json"

    main(output, stenosis_path, vessel_path, mapping)
