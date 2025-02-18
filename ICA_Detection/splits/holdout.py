# ica_yolo_detection/splits/holdout.py

import os
import sys
import json
import random
import math
from typing import Dict, List


def validate_splits(splits: Dict[str, float]) -> bool:
    """
    Validate that the splits dictionary has at least two keys and that the values sum to 1.
    """
    if len(splits) < 2:
        print("Error: At least two splits must be provided.")
        return False
    total = sum(splits.values())
    if not math.isclose(total, 1.0, abs_tol=1e-3):
        print(f"Error: The splits sum to {total}, but should sum to 1.0.")
        return False
    return True


def get_image_files(folder: str) -> List[str]:
    """
    Return a sorted list of image filenames (only files) in the given folder.
    """
    return sorted(
        [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    )


def group_by_patient(file_list: List[str]) -> Dict[str, List[str]]:
    """
    Group filenames by patient. Each filename follows the convention:

        {dataset}_{patient}_{video}_{frame}.ext

    The patient group is defined as: "{dataset}_{patient}".

    Returns:
        Dict[str, List[str]]: Mapping from patient group key to list of filenames.
    """
    groups: Dict[str, List[str]] = {}
    for filename in file_list:
        # Remove extension and split by underscore.
        base, _ = os.path.splitext(filename)
        parts = base.split("_")
        if len(parts) < 4:
            print(f"Skipping file with unexpected format: {filename}")
            continue
        patient_key = f"{parts[0]}_{parts[1]}"
        groups.setdefault(patient_key, []).append(filename)
    return groups


def create_symbolic_links(
    file_list: List[str], src_folder: str, dest_folder: str
) -> None:
    """
    Create symbolic links in dest_folder for each file in file_list located in src_folder.
    Overwrites any existing links.
    """
    for f in file_list:
        src = os.path.join(src_folder, f)
        dst = os.path.join(dest_folder, f)
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)


def create_holdout_split(
    input_root: str,
    splits: Dict[str, float],
    output_root: str,
    splits_info_filename: str = "splits_info.json",
    yaml_filename: str = "dataset.yaml",
) -> None:
    """
    Create a holdout split at the dataset-patient level. All images from the same patient (as defined by
    {dataset}_{patient}) are assigned to the same split.

    Inputs:
      1. input_root: Folder with subfolders "images" and "labels" (all images/labels are in YOLO format).
      2. splits: Dictionary with split names and percentages, e.g., {"train": 0.6, "val": 0.2, "test": 0.2}.
         Must contain at least two splits and sum to 1.

    The function creates the following structure in output_root:

        output_root/
            images/
                train/
                val/
                test/
            labels/
                train/
                val/
                test/
            dataset.yaml
            splits_info.json   # JSON file with assigned patients per split.

    The dataset.yaml file includes the absolute path, relative image folder paths, and class names.
    """
    if not validate_splits(splits):
        sys.exit(1)

    # Define input folders.
    in_images = os.path.join(input_root, "images")
    in_labels = os.path.join(input_root, "labels")
    if not os.path.isdir(in_images) or not os.path.isdir(in_labels):
        print("Error: Input root must contain 'images' and 'labels' subfolders.")
        sys.exit(1)

    # List all image files.
    all_files = get_image_files(in_images)
    # Group by patient.
    patient_groups = group_by_patient(all_files)
    all_patients = list(patient_groups.keys())
    random.shuffle(all_patients)
    n_patients = len(all_patients)
    print(f"Total patient groups: {n_patients}")

    # Compute number of patients per split.
    split_assignments = {}
    start = 0
    for split_name, percentage in splits.items():
        count = int(round(percentage * n_patients))
        split_assignments[split_name] = all_patients[start : start + count]
        start += count
    if start < n_patients:
        # Add remaining patients to the last split.
        last_split = list(splits.keys())[-1]
        split_assignments[last_split].extend(all_patients[start:])

    # Prepare output directories.
    out_images_root = os.path.join(output_root, "images")
    out_labels_root = os.path.join(output_root, "labels")
    os.makedirs(out_images_root, exist_ok=True)
    os.makedirs(out_labels_root, exist_ok=True)

    # Dictionary to store split info per dataset.
    splits_info: Dict[str, Dict[str, List[str]]] = {}
    for split_name in splits.keys():
        splits_info[split_name] = {}

    # For each split, create subdirectories and link files.
    for split_name, patients in split_assignments.items():
        split_img_out = os.path.join(out_images_root, split_name)
        split_lbl_out = os.path.join(out_labels_root, split_name)
        os.makedirs(split_img_out, exist_ok=True)
        os.makedirs(split_lbl_out, exist_ok=True)

        for patient in patients:
            # For each patient group, determine dataset name.
            dataset_name, _ = patient.split("_", 1)
            # Record this patient under the appropriate dataset.
            splits_info[split_name].setdefault(dataset_name, []).append(patient)
            # Get all image files for this patient.
            patient_files = patient_groups[patient]
            # Create symbolic links for each file.
            create_symbolic_links(patient_files, in_images, split_img_out)
            # For labels, assume the label file has same base name with .txt extension.
            label_files = [os.path.splitext(f)[0] + ".txt" for f in patient_files]
            create_symbolic_links(label_files, in_labels, split_lbl_out)
        print(f"Created split '{split_name}' with {len(patients)} patient groups.")

    # Generate YAML file.
    yaml_lines = []
    out_root_abs = os.path.abspath(output_root)
    yaml_lines.append(f"path: {out_root_abs}")
    for split_name in splits.keys():
        yaml_lines.append(f"{split_name}: images/{split_name}")
    yaml_lines.append("names:")
    yaml_lines.append("    0: stenosis")
    yaml_content = "\n".join(yaml_lines)
    yaml_path = os.path.join(output_root, yaml_filename)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"YAML file saved to {yaml_path}")

    # Save splits_info JSON.
    splits_info_path = os.path.join(output_root, splits_info_filename)
    with open(splits_info_path, "w") as f:
        json.dump(splits_info, f, indent=4)
    print(f"Splits info JSON saved to {splits_info_path}")


if __name__ == "__main__":
    input_root = "/home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION"
    splits_dict = {"train": 0.7, "val": 0.3, "test": 0.0}
    output_root = "/home/mariopasc/Python/Datasets/COMBINED/YOLO_ICA_DETECTION"
    yaml_filename = "yolo_ica_detection.yaml"

    create_holdout_split(
        input_root,
        splits_dict,
        output_root,
        yaml_filename=yaml_filename,
        splits_info_filename="splits_info.json",
    )
