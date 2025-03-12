import os
import json
from typing import List, Dict, Tuple, Any
import requests
import zipfile
import shutil

# Import the translator functions for Arcade.
from ICA_Detection.tools.bbox_translation import arcade_to_common


def download_and_extract_arcade(download_url: str, extract_to: str) -> None:
    # (Unchanged download and extraction code)
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
    os.remove(local_zip)
    extracted_folder: str = os.path.join(extract_to, "arcade")
    arcade_folder: str = os.path.join(extract_to, "ARCADE")
    if os.path.exists(extracted_folder):
        os.rename(extracted_folder, arcade_folder)
        print(f"Renamed folder to {arcade_folder}")
    else:
        print(f"Expected folder 'arcade' not found in {extract_to}")


def process_arcade_annotation_file(
    annot_file: str, images_folder: str, split: str, unique_id_counter: int
) -> Tuple[Dict[str, Any], int]:
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
        file_name: str = img["file_name"]  # e.g., "98.png"
        num: str = os.path.splitext(file_name)[0]  # e.g., "98"
        unique_id: str = f"arcade{split}_p{num}_v{num}_{num.zfill(5)}"

        img_width: float = img["width"]
        img_height: float = img["height"]
        image_path: str = os.path.join(images_folder, file_name)

        annots: List[Dict[str, Any]] = image_to_annots.get(img["id"], [])

        ann_dict: Dict[str, Any] = {"name": f"{unique_id}.txt"}

        bbox_count: int = 1
        for a in annots:
            # Store the bboxes
            bbox_list: List[float] = a.get("bbox", [])
            if len(bbox_list) < 4:
                continue
            # Build native arcade bbox dictionary.
            native_bbox = {
                "x": bbox_list[0],
                "y": bbox_list[1],
                "w": bbox_list[2],
                "h": bbox_list[3],
                "label": cat_mapping.get(
                    a.get("category_id"), str(a.get("category_id"))  # type: ignore
                ),
            }
            common_bbox = arcade_to_common(native_bbox)
            ann_dict[f"bbox{bbox_count}"] = common_bbox

    
            # Hi! Uncomment me to store the stenosis segmentation of the ARCADE dataset
            """
            segmentation_data = a.get("segmentation", [])
            # If you need a flattened array
            if segmentation_data and isinstance(segmentation_data, list):
                # If it's a list of lists (polygon format), flatten it
                if segmentation_data and isinstance(segmentation_data[0], list):
                    flattened_segmentation = segmentation_data[
                        0
                    ]  # Take the first polygon
                else:
                    flattened_segmentation = segmentation_data

                native_segmentation = {
                    "xyxy": flattened_segmentation,
                    "label": cat_mapping.get(
                        a.get("category_id"), str(a.get("category_id"))  # type: ignore
                    ),
                }

                ann_dict[f"segmentation{bbox_count}"] = native_segmentation

            bbox_count += 1
            """
        lesion_flag: bool = True if bbox_count > 1 else False

        entry: Dict[str, Any] = {
            "id": unique_id,
            "dataset_origin": "arcade",
            "lesion": lesion_flag,
            "image": {
                "name": f"{unique_id}.png",
                "route": image_path,
                "original_name": file_name,
                "height": img_height,
                "width": img_width,
            },
            "annotations": ann_dict,
        }
        output_entries[unique_id] = entry
    return output_entries, unique_id_counter


def process_arcade_stenosis_dataset(root_dir: str) -> Dict[str, Any]:
    """
    Process the ARCADE dataset for stenosis detection only.
    
    Args:
        root_dir: Root directory containing the ARCADE dataset
        
    Returns:
        Dictionary containing the standardized stenosis detection dataset
    """
    root_dir = os.path.join(root_dir, "ARCADE")
    stenosis_dataset: Dict[str, Any] = {}
    unique_id_counter: int = 1

    stenosis_dir: str = os.path.join(root_dir, "stenosis")
    if os.path.isdir(stenosis_dir):
        for split in ["train", "val", "test"]:
            split_dir: str = os.path.join(stenosis_dir, split)
            if not os.path.isdir(split_dir):
                continue
            images_folder: str = os.path.join(split_dir, "images")
            annotations_folder: str = os.path.join(split_dir, "annotations")
            annot_file: str = os.path.join(annotations_folder, f"{split}.json")
            if not os.path.exists(annot_file):
                print(f"Annotation file not found: {annot_file}")
                continue
            entries, unique_id_counter = process_arcade_annotation_file(
                annot_file, images_folder, split, unique_id_counter
            )
            stenosis_dataset.update(entries)

    return {"Stenosis_Detection": stenosis_dataset}


def process_arcade_vessel_segmentation(root_dir: str) -> Dict[str, Any]:
    """
    Process the ARCADE dataset for vessel segmentation task only.
    
    Args:
        root_dir: Root directory containing the ARCADE dataset
        
    Returns:
        Dictionary containing the standardized vessel segmentation dataset
    """
    root_dir = os.path.join(root_dir, "ARCADE")
    arteries_dataset: Dict[str, Any] = {}
    unique_id_counter: int = 1

    syntax_dir: str = os.path.join(root_dir, "syntax")
    if not os.path.isdir(syntax_dir):
        print(f"Syntax directory not found: {syntax_dir}")
        return {"Arteries_Segmentation": {}}

    image_arteries_path = os.path.join(
        root_dir, "../../ICA_DETECTION", "images", "images_arteries"
    )
    os.makedirs(image_arteries_path, exist_ok=True)

    # Process syntax data by split
    for split in ["train", "val", "test"]:
        split_dir: str = os.path.join(syntax_dir, split)
        if not os.path.isdir(split_dir):
            continue
            
        images_folder: str = os.path.join(split_dir, "images")
        
        # Copy and rename images
        for image in os.listdir(images_folder):
            num = os.path.splitext(image)[0]  # e.g., "98"
            unique_id = f"arcade{split}_p{num}_v{num}_{num.zfill(5)}.png"
            
            # Copy then rename to avoid overwriting
            shutil.copyfile(
                src=os.path.join(images_folder, image),
                dst=os.path.join(image_arteries_path, image),
            )
            os.rename(
                src=os.path.join(image_arteries_path, image),
                dst=os.path.join(image_arteries_path, unique_id)
            )

        annotations_folder: str = os.path.join(split_dir, "annotations")
        annot_file: str = os.path.join(annotations_folder, f"{split}.json")
        if not os.path.exists(annot_file):
            print(f"Syntax annotation file not found: {annot_file}")
            continue

        # Load the syntax JSON file
        with open(annot_file, "r") as f:
            syntax_data = json.load(f)

        syntax_images = syntax_data.get("images", [])
        syntax_annotations = syntax_data.get("annotations", [])
        syntax_categories = syntax_data.get("categories", [])

        # Build category mapping for syntax
        syntax_cat_mapping = {cat["id"]: cat["name"] for cat in syntax_categories}

        # Build mapping from image_id to filename
        id_to_filename = {img["id"]: img["file_name"] for img in syntax_images}
        
        # Map image_id to image dimensions
        id_to_dimensions = {img["id"]: (img["width"], img["height"]) for img in syntax_images}

        # Group annotations by image_id
        img_id_to_annots: Dict[Any, List[Dict[str, Any]]] = {}
        for ann in syntax_annotations:
            img_id = ann["image_id"]
            img_id_to_annots.setdefault(img_id, []).append(ann)

        # Create entries for each image
        for img_id, annots in img_id_to_annots.items():
            filename = id_to_filename.get(img_id)
            if not filename:
                continue
                
            # Get image dimensions
            img_width, img_height = id_to_dimensions.get(img_id, (0, 0))
                
            num = os.path.splitext(filename)[0]  # e.g., "98"
            unique_id = f"arcade{split}_p{num}_v{num}_{num.zfill(5)}"
            
            vessel_segmentations = []
            for i, ann in enumerate(annots):
                cat_id = ann.get("category_id")
                segmentation_data = ann.get("segmentation", [])
                area_data = ann.get("area", 0)
                bbox_data = ann.get("bbox", [])
                iscrowd_data = ann.get("iscrowd", 0)
                attributes = ann.get("attributes", {})

                # Skip if incomplete data
                if len(bbox_data) < 4:
                    continue

                bbox_data_parsed = {
                    "xmin": bbox_data[0],
                    "ymin": bbox_data[1],
                    "xmax": bbox_data[0] + bbox_data[2],  # Convert width to xmax
                    "ymax": bbox_data[1] + bbox_data[3],  # Convert height to ymax
                    "label": cat_id,
                    "category": syntax_cat_mapping.get(cat_id, str(cat_id)),
                }

                if segmentation_data and isinstance(segmentation_data, list):
                    # If it's a list of lists (polygon format), flatten it
                    if segmentation_data and isinstance(segmentation_data[0], list):
                        flattened_segmentation = segmentation_data[0]  # Take the first polygon
                    else:
                        flattened_segmentation = segmentation_data

                    vessel_segmentation = {
                        f"segment{i}": flattened_segmentation,
                        "bbox": bbox_data_parsed,
                        "area": area_data,
                        "iscrowd": iscrowd_data,
                        "attributes": attributes,
                    }
                    vessel_segmentations.append(vessel_segmentation)

            # Create the entry
            ann_dict: Dict[str, Any] = {"name": f"{unique_id}.txt", "vessel_segmentations": vessel_segmentations}
            
            entry: Dict[str, Any] = {
                "id": unique_id,
                "dataset_origin": "arcade",
                "image": {
                    "name": f"{unique_id}.png",
                    "route": os.path.join(image_arteries_path, f"{unique_id}.png"),
                    "original_name": filename,
                    "height": img_height,
                    "width": img_width,
                },
                "annotations": ann_dict,
            }
            
            arteries_dataset[unique_id] = entry

    return {"Arteries_Segmentation": arteries_dataset}


def process_arcade_dataset(root_dir: str, task: str = "both") -> Dict[str, Any]:
    """
    Process the ARCADE dataset based on the specified task.
    
    Args:
        root_dir: Root directory containing the ARCADE dataset
        task: Task to process. Options are "stenosis", "arteries", or "both"
        
    Returns:
        Dictionary containing the standardized dataset(s)
    """
    result: Dict[str, Any] = {}
    
    if task.lower() in ["stenosis", "both"]:
        stenosis_data = process_arcade_stenosis_dataset(root_dir)
        result.update(stenosis_data)
        
    if task.lower() in ["arteries", "both"]:
        arteries_data = process_arcade_vessel_segmentation(root_dir)
        result.update(arteries_data)
        
    return result


if __name__ == "__main__":
    # Example usage:
    download_url: str = (
        "https://data.mendeley.com/public-files/datasets/ydrm75xywg/files/61f788d6-65ce-4265-a23a-5ba16931d18b/file_downloaded"
    )
    extract_folder: str = "/home/mario/Python/Datasets/COMBINED/source"
    # download_and_extract_arcade(download_url, extract_folder)
    
    # Process both tasks
    json_data: Dict[str, Any] = process_arcade_dataset(extract_folder, task="both")
    output_json_file: str = "arcade_standardized.json"
    with open(output_json_file, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Standardized JSON saved to {output_json_file}")