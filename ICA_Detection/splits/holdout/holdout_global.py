import os
import sys
import json
import random
import math
from typing import Dict, List, Optional


def validate_splits(splits: Dict[str, float]) -> bool:
    """
    Validate that the splits dictionary has at least two keys
    and that the values sum (approximately) to 1.0.
    """
    if len(splits) < 2:
        print("Error: At least two splits must be provided.")
        return False
    total = sum(splits.values())
    if not math.isclose(total, 1.0, abs_tol=1e-3):
        print(f"Error: The splits sum to {total}, but should sum to 1.0.")
        return False
    return True


def group_by_patient(file_list: List[str]) -> Dict[str, List[str]]:
    """
    Group filenames by patient. Each filename is expected to follow the
    convention: {dataset}_{patient}_{video}_{frame}.ext

    E.g. "cadica_p26_v1_00020.png"
         => dataset = "cadica", patient = "p26"

    The grouping key is "{dataset}_{patient}", all lowercased.
    """
    groups: Dict[str, List[str]] = {}
    for filename in file_list:
        base, _ = os.path.splitext(filename)
        parts = base.split("_")
        if len(parts) < 4:
            print(f"Skipping file with unexpected format: {filename}")
            continue
        patient_key = f"{parts[0].lower()}_{parts[1].lower()}"  # e.g., "cadica_p26"
        groups.setdefault(patient_key, []).append(filename)
    return groups


def get_image_files(folder: str) -> List[str]:
    """
    Return a sorted list of image filenames in the given folder (only files).
    """
    return sorted(
        [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    )


def holdout(
    images_folder: str,
    splits: Dict[str, float],
    output_json_path: str,
    include_datasets: Optional[List[str]] = None,
    seed: int = 42
) -> None:
    """
    Perform patient-level holdout splitting, then save the results to a JSON file.
    This function DOES NOT create any subfolders or symlinks; it only stores
    which patient groups belong to which split.

    :param images_folder: Path to the folder containing all images.
    :param splits: Dictionary with split names and percentages, e.g. {"train": 0.7, "val": 0.3}.
    :param output_json_path: Where to save the splits info JSON.
    :param include_datasets: If given, filter images by dataset prefix
                             (case-insensitive). E.g. ["CADICA", "KEMEROVO"].
    """
    if not validate_splits(splits):
        sys.exit(1)

    if not os.path.isdir(images_folder):
        print(f"Error: images_folder does not exist: {images_folder}")
        sys.exit(1)

    random.seed(seed)

    # 1) Gather all image filenames
    all_files = get_image_files(images_folder)

    # 2) Filter by dataset prefix if requested
    if include_datasets:
        allowed_prefixes = [ds.lower() for ds in include_datasets]
        # Special handling for "ARCADE" if needed
        if "ARCADE" in include_datasets:
            allowed_prefixes.extend(["arcadetrain", "arcadeval", "arcadetest"])

        filtered = []
        for f in all_files:
            base, _ = os.path.splitext(f)
            prefix = base.split("_")[0].lower()
            if prefix in allowed_prefixes:
                filtered.append(f)
        all_files = filtered

    # 3) Group by patient
    patient_groups = group_by_patient(all_files)
    all_patients = list(patient_groups.keys())
    random.shuffle(all_patients)

    n_patients = len(all_patients)
    print(f"Total patient groups after filtering: {n_patients}")

    # 4) Assign patient groups to splits
    assigned: Dict[str, List[str]] = {s: [] for s in splits.keys()}
    start_idx = 0
    for split_name, percentage in splits.items():
        count = int(round(percentage * n_patients))
        assigned[split_name] = all_patients[start_idx : start_idx + count]
        start_idx += count

    # If there's a remainder due to rounding, put them in the last split
    if start_idx < n_patients:
        last_split = list(splits.keys())[-1]
        assigned[last_split].extend(all_patients[start_idx:])

    # 5) Build the final dictionary structure
    #    e.g. {
    #        "train": {
    #            "cadica": ["cadica_p26", "cadica_p28"],
    #            "kemerovo": ["kemerovo_p13"],
    #            ...
    #        },
    #        "val": {...},
    #        ...
    #    }
    splits_info: Dict[str, Dict[str, List[str]]] = {}
    for split_name in assigned:
        splits_info[split_name] = {}
        for patient_key in assigned[split_name]:
            dataset_prefix, _ = patient_key.split("_", 1)  # e.g. "cadica", "p26"
            splits_info[split_name].setdefault(dataset_prefix, []).append(patient_key)

    # 6) Save splits_info JSON
    with open(output_json_path, "w") as f:  # type: ignore
        json.dump(splits_info, f, indent=2)  # type: ignore

    print(f"Splits info JSON saved to {output_json_path}")


if __name__ == "__main__":
    # Example usage
    random.seed(42)
    images_path = "/path/to/ICA_DETECTION/images"
    splits_dict = {"train": 0.7, "val": 0.3, "test": 0.0}
    output_json = "splits_info.json"

    holdout(
        images_folder=images_path,
        splits=splits_dict,
        output_json_path=output_json,
        include_datasets=["CADICA"],  # or None if you want all
    )
