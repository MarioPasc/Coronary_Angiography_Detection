# ica_yolo_detection/integration/integrate.py

import json
from typing import List, Dict, Any

# Import the dataset-specific processing modules.
from ICA_Detection.integration.arcade import process_arcade_dataset
from ICA_Detection.integration.cadica import process_cadica_dataset
from ICA_Detection.integration.kemerovo import process_kemerovo_dataset


def integrate_datasets(
    datasets: List[str], root_dirs: Dict[str, str]
) -> Dict[str, Any]:
    """
    Integrate standardized JSON outputs from multiple datasets.

    Args:
        datasets (List[str]): List of dataset names to process. Valid names are
            "CADICA", "ARCADE", and "KEMEROVO". For example: ["CADICA", "ARCADE", "KEMEROVO"].
        root_dirs (Dict[str, str]): Mapping from dataset name to its root directory.
            For example:
                {
                    "CADICA": "/home/mario/Python/Datasets",
                    "ARCADE": "/home/mario/Python/Datasets",
                    "KEMEROVO": "/home/mario/Python/Datasets"
                }
        arcade_task (str, optional): For ARCADE, which task to process ("stenosis", "syntax", or "both"). Defaults to "stenosis".

    Returns:
        Dict[str, Any]: A combined JSON dictionary with a top-level key "Standard_dataset"
        that contains entries from all selected datasets.
    """
    combined_entries: Dict[str, Any] = {}

    for ds in datasets:
        ds_upper = ds.upper()
        if ds_upper == "CADICA":
            root = root_dirs.get("CADICA")
            if not root:
                print("Root directory for CADICA not provided. Skipping CADICA.")
                continue
            print("Processing CADICA dataset...")
            data = process_cadica_dataset(root)
            combined_entries.update(data.get("Stenosis_Detection", {}))
        elif ds_upper == "ARCADE":
            root = root_dirs.get("ARCADE")
            if not root:
                print("Root directory for ARCADE not provided. Skipping ARCADE.")
                continue
            print("Processing ARCADE dataset...")
            # process_arcade_dataset expects the folder that contains the "ARCADE" folder.
            data = process_arcade_dataset(root, task='stenosis')
            combined_entries.update(data.get("Stenosis_Detection", {}))
            
            data_segmentation = process_arcade_dataset(root, task='arteries')
            
        elif ds_upper == "KEMEROVO":
            root = root_dirs.get("KEMEROVO")
            if not root:
                print("Root directory for KEMEROVO not provided. Skipping KEMEROVO.")
                continue
            print("Processing KEMEROVO dataset...")
            data = process_kemerovo_dataset(root)
            combined_entries.update(data.get("Stenosis_Detection", {}))
        else:
            print(f"Dataset '{ds}' not recognized. Skipping.")

    return {"detection": {"Stenosis_Detection": combined_entries},
            "segmentation": {"Arteries_Segmentation":data_segmentation}}


if __name__ == "__main__":
    # You can download the datasets manually from:
    # - KEMEROV: https://data.mendeley.com/datasets/ydrm75xywg/2
    # - ARCADE: https://zenodo.org/records/10390295
    # - CADICA: https://data.mendeley.com/datasets/p9bpx9ctcv/2

    # Define the list of datasets to integrate.
    datasets_to_process = ["CADICA", "ARCADE", "KEMEROVO"]

    # Provide a mapping from dataset names to the required root directories.
    # Note: For CADICA and ARCADE, use the root folders expected by their processing functions.
    root_dirs = {
        "CADICA": "/home/mario/Python/Datasets/COMBINED/source",
        "ARCADE": "/home/mario/Python/Datasets/COMBINED/source",
        "KEMEROVO": "/home/mario/Python/Datasets/COMBINED/source",
    }

    # Integrate the selected datasets.
    final_json: Dict[str, Any] = integrate_datasets(datasets_to_process, root_dirs)

    # Save the combined JSON to a file.
    output_file = "combined_standardized.json"
    with open(output_file, "w") as f:
        json.dump(final_json, f, indent=4)
    print(f"Combined standardized JSON saved to {output_file}")
