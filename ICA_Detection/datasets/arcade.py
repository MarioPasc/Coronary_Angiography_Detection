# ica_yolo_detection/datasets/dataset_arcade.py

import os
import json
from typing import List, Dict, Tuple, Any


def convert_bbox_yolo(
    x: float, y: float, w: float, h: float, img_width: float, img_height: float
) -> Dict[str, float]:
    """
    Convert bounding box from XYWH (pixel, top-left based) to YOLO normalized format.

    Args:
        x (float): x coordinate of the top-left corner (in pixels).
        y (float): y coordinate of the top-left corner (in pixels).
        w (float): width of the box (in pixels).
        h (float): height of the box (in pixels).
        img_width (float): width of the image (in pixels).
        img_height (float): height of the image (in pixels).

    Returns:
        Dict[str, float]: Dictionary with keys "x_center", "y_center", "width", "height" (all normalized).
    """
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    return {
        "x_center": x_center,
        "y_center": y_center,
        "width": w / img_width,
        "height": h / img_height,
    }


def process_arcade_annotation_file(
    annot_file: str, images_folder: str, unique_id_start: int
) -> Tuple[Dict[str, Any], int]:
    """
    Process a single ARCADE annotation JSON file.

    The ARCADE annotation JSON contains three top-level fields:
      - "images": list of image dictionaries (each with "id", "width", "height", "file_name").
      - "categories": list of category dictionaries (each with "id" and "name").
      - "annotations": list of annotation dictionaries (each with "id", "image_id", "category_id", "bbox", etc.).

    For each image, we find its matching annotations (if any), convert each bounding box from pixel XYWH to YOLO normalized format,
    and create a standardized entry.

    Args:
        annot_file (str): Path to the ARCADE annotation JSON file.
        images_folder (str): Path to the folder containing the image files.
        unique_id_start (int): Starting counter for standardized image IDs.

    Returns:
        Tuple[Dict[str, Any], int]: A tuple containing:
           - A dictionary mapping standardized IDs (e.g., "arcade_1") to JSON entries.
           - The updated unique_id counter.
    """
    with open(annot_file, "r") as f:
        data = json.load(f)

    images_list: List[Dict[str, Any]] = data.get("images", [])
    annotations_list: List[Dict[str, Any]] = data.get("annotations", [])
    categories_list: List[Dict[str, Any]] = data.get("categories", [])

    # Build mapping from category_id to category name.
    cat_mapping: Dict[int, str] = {cat["id"]: cat["name"] for cat in categories_list}

    # Build mapping from image_id to list of its annotations.
    image_to_annots: Dict[Any, List[Dict[str, Any]]] = {}
    for ann in annotations_list:
        img_id = ann["image_id"]
        image_to_annots.setdefault(img_id, []).append(ann)

    output_entries: Dict[str, Any] = {}
    for img in images_list:
        img_id = img["id"]
        file_name = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        image_path = os.path.join(images_folder, file_name)

        # Get matching annotations (if any).
        annots: List[Dict[str, Any]] = image_to_annots.get(img_id, [])

        # Build the annotations dictionary.
        ann_dict: Dict[str, Any] = {"name": f"arcade_{unique_id_start}.txt"}
        bbox_count = 1
        for a in annots:
            bbox = a.get("bbox", [])  # bbox in [x, y, w, h] format
            if len(bbox) < 4:
                continue
            converted = convert_bbox_yolo(
                bbox[0], bbox[1], bbox[2], bbox[3], img_width, img_height
            )
            # Look up the category name using the category_id.
            category_id = a.get("category_id")
            label = cat_mapping.get(category_id, str(category_id))  # type: ignore
            converted["label"] = label  # type: ignore
            ann_dict[f"bbox{bbox_count}"] = converted
            bbox_count += 1

        # If there is at least one bounding box, mark lesion as True.
        lesion_flag = True if bbox_count > 1 else False

        std_id = f"arcade_{unique_id_start}"
        entry = {
            "id": std_id,
            "dataset_origin": "arcade",
            "lesion": lesion_flag,
            "image": {
                "name": f"{std_id}.png",
                "route": image_path,
                "original_name": file_name,
                "height": img_height,
                "width": img_width,
            },
            "annotations": ann_dict,
        }
        output_entries[std_id] = entry
        unique_id_start += 1

    return output_entries, unique_id_start


def process_arcade_dataset(root_dir: str) -> Dict[str, Any]:
    """
    Process the ARCADE dataset given its root directory.

    The ARCADE dataset directory structure is as follows:

        root_dir/
            stenosis/
                train/
                    images/
                    annotations/train.json
                val/
                    images/
                    annotations/val.json
                test/
                    images/
                    annotations/test.json
            syntax/
                train/
                    images/
                    annotations/train.json
                val/
                    images/
                    annotations/val.json
                test/
                    images/
                    annotations/test.json

    This function processes every split (train, val, test) for both modalities ("stenosis" and "syntax"),
    merging all entries into one standardized JSON dictionary.

    Args:
        root_dir (str): Path to the ARCADE dataset root directory (e.g., "/home/mariopasc/Python/Datasets/arcade").

    Returns:
        Dict[str, Any]: A dictionary with the standardized JSON structure:
           {
             "Standard_dataset": {
                 "arcade_1": { ... },
                 "arcade_2": { ... },
                 ...
             }
           }
    """
    standard_dataset: Dict[str, Any] = {}
    unique_id_counter: int = 1
    # Process both modalities.
    for modality in ["stenosis", "syntax"]:
        modality_dir = os.path.join(root_dir, modality)
        if not os.path.isdir(modality_dir):
            continue
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(modality_dir, split)
            if not os.path.isdir(split_dir):
                continue
            images_folder = os.path.join(split_dir, "images")
            annotations_folder = os.path.join(split_dir, "annotations")
            annot_file = os.path.join(annotations_folder, f"{split}.json")
            if not os.path.exists(annot_file):
                print(f"Annotation file not found: {annot_file}")
                continue
            entries, unique_id_counter = process_arcade_annotation_file(
                annot_file, images_folder, unique_id_counter
            )
            standard_dataset.update(entries)
    return {"Standard_dataset": standard_dataset}


if __name__ == "__main__":
    # Example usage:
    # Set the root directory for the ARCADE dataset.
    root_dir: str = "/home/mariopasc/Python/Datasets/arcade"
    json_data: Dict[str, Any] = process_arcade_dataset(root_dir)
    output_json_file: str = "./arcade_standardized.json"
    with open(output_json_file, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Standardized JSON saved to {output_json_file}")
