import os
import json
from pathlib import Path
from typing import Dict, Any, Union, Tuple
import shutil
from PIL import Image

from .bbox_translation import calculate_polygon_area, calculate_bbox_from_segmentation


def construct_yolo(root_folder: Union[str, Path]) -> None:
    """
    Create the YOLO dataset folder structure under root_folder/datasets/yolo
    by:
      1. Creating the subfolders: images/, labels/
      2. Making symbolic links in images/ to all files under root_folder/images
      3. Copying (or renaming) labels_yolo/ to labels/

    :param root_folder: Path to the root folder containing images/, labels_yolo/, etc.
    """
    root_path = Path(root_folder).resolve()
    yolo_path = root_path / "datasets" / "yolo"
    yolo_images = yolo_path / "images"
    yolo_labels = yolo_path / "labels"

    # Ensure the yolo/ directory structure exists
    yolo_images.mkdir(parents=True, exist_ok=True)
    yolo_labels.mkdir(parents=True, exist_ok=True)

    # 1) Create symbolic links for images
    original_images_dir = root_path / "images"
    for image_file in original_images_dir.iterdir():
        if image_file.is_file():
            # e.g., "cadica_p26_v1_00020.png"
            link_dest = yolo_images / image_file.name
            # os.symlink(src=..., dst=...) must not exist at dst
            if link_dest.exists():
                link_dest.unlink()
            os.symlink(image_file, link_dest)

    # 2) Copy YOLO labels
    original_labels_yolo = root_path / "labels_yolo"
    if original_labels_yolo.exists() and original_labels_yolo.is_dir():
        for label_file in original_labels_yolo.iterdir():
            if label_file.is_file():
                shutil.copy(label_file, yolo_labels / label_file.name)


def construct_coco_compatible(
    json_path: Union[str, Path], root_folder: Union[str, Path]
) -> Tuple[str, str]:
    """
    Creates a dataset folder (if needed) and three JSON files:
      1) The original "Standard_dataset" JSON, with possibly updated paths
      2) A COCO-style JSON for stenosis data, to enable COCO evaluation with CocoEvaluator
      3) A COCO-style JSON for arteries data, containing vessel segmentations

    Returns the paths to the newly created COCO JSONs (stenosis_coco.json, arteries_coco.json).
    """

    root_path = Path(root_folder).resolve()
    datasets_path = root_path / "datasets"
    datasets_path.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    # -----------------------------------------------------------------
    # 1) Re-write the "Standard_dataset"
    # -----------------------------------------------------------------
    new_data: Dict[str, Any] = {"Standard_dataset": {}}
    for key, item in data["Standard_dataset"].items():
        original_img_path = item["image"]["dataset_route"]

        new_item = dict(item)
        new_item["image"] = dict(item["image"])
        new_item["image"]["dataset_route"] = original_img_path

        new_data["Standard_dataset"][key] = new_item

    # -----------------------------------------------------------------
    # 2) Create a COCO-style JSON for stenosis evaluation
    # -----------------------------------------------------------------
    stenosis_images = []
    stenosis_annotations = []
    stenosis_categories = [{"id": 1, "name": "stenosis"}]

    stenosis_ann_id_counter = 1
    image_id_counter = 1

    for key, item in new_data["Standard_dataset"].items():
        img_path = item["image"]["dataset_route"]

        # Load the image to get width/height
        with Image.open(img_path) as im:
            width, height = im.size

        # Add an entry to the "images" list
        image_id = image_id_counter
        image_id_counter += 1

        stenosis_images.append(
            {
                "id": image_id,
                "file_name": os.path.basename(img_path),
                "width": width,
                "height": height,
            }
        )

        # Process stenosis annotations if they exist
        if "annotations" in item and "stenosis" in item["annotations"]:
            stenosis_data = item["annotations"]["stenosis"]

            # Process bounding boxes
            for ann_key, ann_val in stenosis_data.items():
                if ann_key.startswith("bbox"):
                    xmin = float(ann_val["xmin"])
                    ymin = float(ann_val["ymin"])
                    xmax = float(ann_val["xmax"])
                    ymax = float(ann_val["ymax"])

                    w = xmax - xmin
                    h = ymax - ymin
                    area = w * h
                    if w <= 0 or h <= 0:
                        # Skip invalid boxes
                        continue

                    # Add bounding box annotation
                    stenosis_annotations.append(
                        {
                            "id": stenosis_ann_id_counter,
                            "image_id": image_id,
                            "category_id": 1,  # 'stenosis'
                            "bbox": [xmin, ymin, w, h],  # XYWH format for COCO
                            "area": area,
                            "iscrowd": 0,
                        }
                    )
                    stenosis_ann_id_counter += 1

                # Process segmentation data
                elif ann_key.startswith("segmentation"):
                    # COCO segmentation format requires [x1,y1,x2,y2,...] format
                    seg_points = ann_val["xyxy"]

                    # Calculate area of the polygon
                    # Simple approach: use polygon area formula
                    area = calculate_polygon_area(seg_points)

                    stenosis_annotations.append(
                        {
                            "id": stenosis_ann_id_counter,
                            "image_id": image_id,
                            "category_id": 1,  # 'stenosis'
                            "segmentation": [
                                seg_points
                            ],  # COCO expects list of polygons
                            "area": area,
                            "iscrowd": 0,
                            "bbox": calculate_bbox_from_segmentation(
                                seg_points
                            ),  # Required for COCO
                        }
                    )
                    stenosis_ann_id_counter += 1

    # Build stenosis COCO dict
    stenosis_coco_dict = {
        "images": stenosis_images,
        "annotations": stenosis_annotations,
        "categories": stenosis_categories,
    }
    os.makedirs(os.path.join(datasets_path, "stenosis"), exist_ok=True)

    # Save stenosis_coco.json
    stenosis_json_path = os.path.join(datasets_path, "stenosis", "stenosis_coco.json")
    with open(stenosis_json_path, "w") as f_stenosis:
        json.dump(stenosis_coco_dict, f_stenosis, indent=2)

    print(f"Stenosis COCO-style annotation saved to: {stenosis_json_path}")

    # -----------------------------------------------------------------
    # 3) Create a COCO-style JSON for arteries evaluation
    # -----------------------------------------------------------------
    arteries_images = stenosis_images.copy()  # Reuse the same images
    arteries_annotations = []
    arteries_categories = [{"id": 1, "name": "artery"}]

    arteries_ann_id_counter = 1

    for key, item in new_data["Standard_dataset"].items():
        # Use the same image_id mapping as before
        image_id = stenosis_images[
            list(new_data["Standard_dataset"].keys()).index(key)
        ]["id"]

        # Process vessel segmentations if they exist
        if "annotations" in item and "vessel_segmentations" in item["annotations"]:
            vessel_segments = item["annotations"]["vessel_segmentations"]

            # Skip if no segments
            if not vessel_segments:
                continue

            # Collect all segmentation points for this image
            all_segmentations = []

            # Find global bounding box (min x, min y, max x, max y) across all segments
            global_x_min = float("inf")
            global_y_min = float("inf")
            global_x_max = float("-inf")
            global_y_max = float("-inf")

            total_area = 0

            # Process each segment
            for segment in vessel_segments:
                # Extract segmentation points
                seg_points = segment["xyxy"]
                all_segmentations.extend(seg_points)

                # Update global bounding box
                x_coords = seg_points[0::2]
                y_coords = seg_points[1::2]

                global_x_min = min(global_x_min, min(x_coords))
                global_y_min = min(global_y_min, min(y_coords))
                global_x_max = max(global_x_max, max(x_coords))
                global_y_max = max(global_y_max, max(y_coords))

                # Add to total area (we'll later use the bounding box area)
                if "area" in segment:
                    total_area += segment["area"]

            # Calculate bounding box in COCO format [x, y, width, height]
            bbox_width = global_x_max - global_x_min
            bbox_height = global_y_max - global_y_min

            # Handle edge case of invalid dimensions
            if bbox_width <= 0 or bbox_height <= 0:
                continue

            # Use bounding box area if total_area is not calculated
            if total_area == 0:
                total_area = bbox_width * bbox_height  # type: ignore

            # Create a single annotation with all segmentations for this image
            arteries_annotations.append(
                {
                    "id": arteries_ann_id_counter,
                    "image_id": image_id,
                    "category_id": 1,  # 'artery'
                    "segmentation": all_segmentations,  # List of all segmentation polygons
                    "bbox": [global_x_min, global_y_min, bbox_width, bbox_height],
                    "area": total_area,
                    "iscrowd": 0,
                }
            )
            arteries_ann_id_counter += 1

    # Build arteries COCO dict
    arteries_coco_dict = {
        "images": arteries_images,
        "annotations": arteries_annotations,
        "categories": arteries_categories,
    }

    os.makedirs(os.path.join(datasets_path, "arteries"), exist_ok=True)
    # Save arteries_coco.json
    arteries_json_path = os.path.join(datasets_path, "arteries", "arteries_coco.json")
    with open(arteries_json_path, "w") as f_arteries:
        json.dump(arteries_coco_dict, f_arteries, indent=2)

    print(f"Arteries COCO-style annotation saved to: {arteries_json_path}")

    return stenosis_json_path, arteries_json_path
