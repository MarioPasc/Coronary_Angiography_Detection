# ica_yolo_detection/datasets/dataset_arcade.py

import os
import json
from typing import List, Dict, Tuple, Any
import requests
import zipfile

def download_and_extract_arcade(download_url: str, extract_to: str) -> None:
    """
    Download the ARCADE dataset zip file from the provided URL, extract it to the specified folder,
    rename the extracted folder "arcade" to "ARCADE", and remove extraneous files/folders.
    
    Args:
        download_url (str): URL to download the arcade zip file.
        extract_to (str): Directory where the dataset should be extracted.
    """
    local_zip: str = os.path.join(extract_to, "arcade_dataset.zip")
    print(f"Downloading ARCADE dataset from {download_url} ...")
    response = requests.get(download_url, stream=True)
    response.raise_for_status()
    with open(local_zip, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded dataset to {local_zip}")

    print(f"Extracting {local_zip} ...")
    with zipfile.ZipFile(local_zip, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted dataset to {extract_to}")

    # Remove the zip file after extraction.
    os.remove(local_zip)

    # The extracted folder is named "arcade"; rename it to "ARCADE".
    extracted_folder: str = os.path.join(extract_to, "arcade")
    arcade_folder: str = os.path.join(extract_to, "ARCADE")
    if os.path.exists(extracted_folder):
        os.rename(extracted_folder, arcade_folder)
        print(f"Renamed folder to {arcade_folder}")
    else:
        print(f"Expected folder 'Stenosis detection' not found in {extract_to}")

def convert_bbox_yolo(x: float, y: float, w: float, h: float, img_width: float, img_height: float) -> Dict[str, float]:
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
        "height": h / img_height
    }

def process_arcade_annotation_file(annot_file: str, images_folder: str, unique_id_start: int) -> Tuple[Dict[str, Any], int]:
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
            converted = convert_bbox_yolo(bbox[0], bbox[1], bbox[2], bbox[3], img_width, img_height)
            # Look up the category name using the category_id.
            category_id = a.get("category_id")
            label = cat_mapping.get(category_id, str(category_id))
            converted["label"] = label
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
                "width": img_width
            },
            "annotations": ann_dict
        }
        output_entries[std_id] = entry
        unique_id_start += 1
        
    return output_entries, unique_id_start

def process_arcade_dataset(root_dir: str, task: str = "stenosis") -> Dict[str, Any]:
    """
    Process the ARCADE dataset given its root directory and the desired task.
    
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
    
    The "task" parameter determines which modality to process:
      - If task == "stenosis": process only the "stenosis" folder.
      - If task == "syntax": process only the "syntax" folder.
      - If task == "both": process both modalities.
    
    Args:
        root_dir (str): Path to the ARCADE dataset root directory (e.g., "/home/mariopasc/Python/Datasets/arcade").
        task (str, optional): Which task to process ("stenosis", "syntax", or "both"). Defaults to "stenosis".
    
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
    root_dir = os.path.join(root_dir, "ARCADE")
    standard_dataset: Dict[str, Any] = {}
    unique_id_counter: int = 1

    # Determine modalities to process based on the task parameter.
    if task == "both":
        modalities: List[str] = ["stenosis", "syntax"]
    elif task in ["stenosis", "syntax"]:
        modalities = [task]
    else:
        print("Invalid task provided; defaulting to 'stenosis'.")
        modalities = ["stenosis"]

    # Process the selected modalities and splits.
    for modality in modalities:
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
            entries, unique_id_counter = process_arcade_annotation_file(annot_file, images_folder, unique_id_counter)
            standard_dataset.update(entries)
    return {"Standard_dataset": standard_dataset}

if __name__ == "__main__":
    # 1. Download and extract the dataset.
    download_url: str = (
       "https://data.mendeley.com/public-files/datasets/ydrm75xywg/files/61f788d6-65ce-4265-a23a-5ba16931d18b/file_downloaded"
    )
    extract_folder: str = (
        "/home/mario/Python/Datasets"  # Adjusted folder path as required
    )
    # download_and_extract_kemerovo(download_url, extract_folder)

    # 2. Process the dataset.  
    # Set the root directory for the ARCADE dataset.
    root_dir: str = "/home/mariopasc/Python/Datasets/arcade"
    # Change the task parameter as needed: "stenosis", "syntax", or "both"
    json_data: Dict[str, Any] = process_arcade_dataset(extract_folder, task="stenosis")
    output_json_file: str = "arcade_standardized.json"
    with open(output_json_file, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Standardized JSON saved to {output_json_file}")
