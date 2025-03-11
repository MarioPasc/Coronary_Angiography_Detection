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
        stenosis_dict: Dict[str, Any] = {}

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
            stenosis_dict[f"bbox{bbox_count}"] = common_bbox

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

                stenosis_dict[f"segmentation{bbox_count}"] = native_segmentation

            bbox_count += 1

        ann_dict["stenosis"] = stenosis_dict

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


def process_arcade_dataset(
    root_dir: str, include_syntax: bool = True
) -> Dict[str, Any]:
    root_dir = os.path.join(root_dir, "ARCADE")
    standard_dataset: Dict[str, Any] = {}
    unique_id_counter: int = 1

    # First process stenosis data (always processed)
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
            standard_dataset.update(entries)

    # If include_syntax is True, process syntax data and add to stenosis entries
    if include_syntax:
        syntax_dir: str = os.path.join(root_dir, "syntax")
        if os.path.isdir(syntax_dir):
            syntax_data_by_split = {}

            # Since the ARCADE dataset does not share ID's between the stenosis
            # and the syntax task, we have to create two different folders that share the same
            # images but have different IDs.
            # For example, an annotation on syntax could refer to ID 656 in train, but ID 656 in train
            # is not the same image for the stenosis task.
            # (sigh)

            # FIXME: We have to fin a way to standarize the name of the images and masks. We may have to entirely chang the json creation

            image_arteries_path = os.path.join(
                root_dir, "../../ICA_DETECTION", "images", "images_arteries"
            )
            os.makedirs(image_arteries_path, exist_ok=True)

            # First load all the syntax data by split
            for split in ["train", "val", "test"]:
                split_dir: str = os.path.join(syntax_dir, split)  # type: ignore
                if not os.path.isdir(split_dir):
                    continue
                images_folder: str = os.path.join(split_dir, "images")  # type: ignore

                for image in os.listdir(images_folder):
                    shutil.copyfile(
                        src=os.path.join(images_folder, image),
                        dst=os.path.join(image_arteries_path, image),
                    )

                annotations_folder: str = os.path.join(split_dir, "annotations")  # type: ignore
                annot_file: str = os.path.join(annotations_folder, f"{split}.json")  # type: ignore
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
                syntax_cat_mapping = {
                    cat["id"]: cat["name"] for cat in syntax_categories
                }

                # Build mapping from image filename to annotations
                syntax_file_to_annots = {}

                # First map image_id to filename
                syntax_id_to_filename = {
                    img["id"]: img["file_name"] for img in syntax_images
                }

                # Then map image_id to annotations
                syntax_img_id_to_annots: Dict[str, Any] = {}
                for ann in syntax_annotations:
                    img_id = ann["image_id"]
                    syntax_img_id_to_annots.setdefault(img_id, []).append(ann)

                # Finally map filename to annotations
                for img_id, annots in syntax_img_id_to_annots.items():
                    filename = syntax_id_to_filename.get(img_id)
                    if filename:
                        num = os.path.splitext(filename)[0]  # e.g., "98"
                        unique_id = f"arcade{split}_p{num}_v{num}_{num.zfill(5)}"
                        syntax_file_to_annots[unique_id] = {
                            "annotations": annots,
                            "categories": syntax_cat_mapping,
                        }

                syntax_data_by_split[split] = syntax_file_to_annots

            # Now add syntax data to stenosis entries
            for unique_id, entry in standard_dataset.items():
                # Parse the unique_id to find the split
                split_part = unique_id.split("_")[0]
                split = split_part.replace("arcade", "")

                # Check if we have syntax data for this split and image
                if (
                    split in syntax_data_by_split
                    and unique_id in syntax_data_by_split[split]
                ):
                    syntax_info = syntax_data_by_split[split][unique_id]

                    # Create vessel_segmentations list
                    vessel_segmentations = []
                    for ann in syntax_info["annotations"]:
                        cat_id = ann.get("category_id")
                        segmentation_data = ann.get("segmentation", [])
                        area_data = ann.get("area", [])
                        bbox_data = ann.get("bbox", [])
                        iscrowd_data = ann.get("iscrowd", [])
                        attributes = ann.get("attributes", [])

                        bbox_data_parsed = {
                            "xmin": bbox_data[0],
                            "ymin": bbox_data[1],
                            "xmax": bbox_data[2],
                            "ymax": bbox_data[3],
                            "label": ann.get("category_id", 0),
                        }

                        if segmentation_data and isinstance(segmentation_data, list):
                            # If it's a list of lists (polygon format), flatten it
                            if segmentation_data and isinstance(
                                segmentation_data[0], list
                            ):
                                flattened_segmentation = segmentation_data[
                                    0
                                ]  # Take the first polygon
                            else:
                                flattened_segmentation = segmentation_data

                            vessel_segmentation = {
                                "xyxy": flattened_segmentation,
                                "bbox": bbox_data_parsed,
                                "area": area_data,
                                "iscrowd": iscrowd_data,
                                "attributes": attributes,
                            }
                            vessel_segmentations.append(vessel_segmentation)

                    # Add vessel_segmentations to entry annotations
                    if vessel_segmentations:
                        entry["annotations"][
                            "vessel_segmentations"
                        ] = vessel_segmentations

    return {"Standard_dataset": standard_dataset}


if __name__ == "__main__":
    # Example usage:
    download_url: str = (
        "https://data.mendeley.com/public-files/datasets/ydrm75xywg/files/61f788d6-65ce-4265-a23a-5ba16931d18b/file_downloaded"
    )
    extract_folder: str = "/home/mariopasc/Python/Datasets/COMBINED/source"
    # download_and_extract_arcade(download_url, extract_folder)
    # json_data: Dict[str, Any] = process_arcade_dataset(extract_folder, task="stenosis")
    json_data: Dict[str, Any] = process_arcade_dataset(extract_folder)
    output_json_file: str = "arcade_standardized.json"
    with open(output_json_file, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Standardized JSON saved to {output_json_file}")
