import os
import json
from typing import List, Dict, Tuple, Any
import requests
import zipfile

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
            bbox_count += 1

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


def process_arcade_dataset(root_dir: str, task: str = "stenosis") -> Dict[str, Any]:
    root_dir = os.path.join(root_dir, "ARCADE")
    standard_dataset: Dict[str, Any] = {}
    unique_id_counter: int = 1

    if task == "both":
        modalities: List[str] = ["stenosis", "syntax"]
    elif task in ["stenosis", "syntax"]:
        modalities = [task]
    else:
        print("Invalid task provided; defaulting to 'stenosis'.")
        modalities = ["stenosis"]

    for modality in modalities:
        modality_dir: str = os.path.join(root_dir, modality)
        if not os.path.isdir(modality_dir):
            continue
        for split in ["train", "val", "test"]:
            split_dir: str = os.path.join(modality_dir, split)
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
            standard_dataset.update(entries)
    return {"Standard_dataset": standard_dataset}


if __name__ == "__main__":
    # Example usage:
    download_url: str = (
        "https://data.mendeley.com/public-files/datasets/ydrm75xywg/files/61f788d6-65ce-4265-a23a-5ba16931d18b/file_downloaded"
    )
    extract_folder: str = "/home/mario/Python/Datasets"
    # download_and_extract_arcade(download_url, extract_folder)
    json_data: Dict[str, Any] = process_arcade_dataset(extract_folder, task="stenosis")
    output_json_file: str = "arcade_standardized.json"
    with open(output_json_file, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Standardized JSON saved to {output_json_file}")
