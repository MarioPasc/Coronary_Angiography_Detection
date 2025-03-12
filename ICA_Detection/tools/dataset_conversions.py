import os
import json
from pathlib import Path
from typing import List, Union
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
    original_labels_yolo = root_path / "labels" / "yolo"
    if original_labels_yolo.exists() and original_labels_yolo.is_dir():
        for label_file in original_labels_yolo.iterdir():
            if label_file.is_file():
                link_dest = yolo_labels / label_file.name
                if link_dest.exists():
                    link_dest.unlink()
                os.symlink(label_file, link_dest)


###########################
# Helper functions
###########################
def calculate_polygon_area(points: List[float]) -> float:
    """
    Calculate the area of a polygon given by a flat list of points:
      [x1, y1, x2, y2, ..., xN, yN]

    This uses the Shoelace formula to compute the area.
    """
    if len(points) < 6:  # At least 3 points (x1,y1,x2,y2,x3,y3)
        return 0.0

    x_coords = points[0::2]
    y_coords = points[1::2]

    n = len(x_coords)
    area = 0.0

    for i in range(n):
        j = (i + 1) % n
        area += x_coords[i] * y_coords[j] - x_coords[j] * y_coords[i]

    return abs(area) / 2.0


def calculate_bbox_from_segmentation(points: List[float]) -> List[float]:
    """
    Given a flat list of points [x1, y1, x2, y2, ..., xN, yN],
    compute the bounding box in [xmin, ymin, width, height] (COCO) format.
    """
    if len(points) < 2:
        return [0, 0, 0, 0]

    x_coords = points[0::2]
    y_coords = points[1::2]

    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)

    return [xmin, ymin, xmax - xmin, ymax - ymin]


###########################
# Detection (Stenosis)
###########################
def construct_coco_detection(
    json_path: Union[str, Path], root_folder: Union[str, Path]
) -> str:
    """
    Build a COCO-style annotation for the "Stenosis_Detection" portion
    of your dataset and save it as 'stenosis_coco.json' in:
        <root_folder>/datasets/stenosis/

    Returns:
        str: The path to the newly created 'stenosis_coco.json' file.
    """
    root_path = Path(root_folder).resolve()
    datasets_path = root_path / "datasets" / "coco"
    datasets_path.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    if "Stenosis_Detection" not in data:
        raise KeyError("JSON does not contain 'Stenosis_Detection' at its top level.")

    # Data subset for detection
    stenosis_data = data["Stenosis_Detection"]

    # Prepare COCO-style lists
    stenosis_images = []
    stenosis_annotations = []
    stenosis_categories = [{"id": 1, "name": "stenosis"}]  # Single class

    stenosis_ann_id_counter = 1
    image_id_counter = 1

    # Iterate over each item in 'Stenosis_Detection'
    for key, item in stenosis_data.items():
        img_path = item["image"]["dataset_route"]

        # Use PIL to read image dimensions
        with Image.open(img_path) as im:
            width, height = im.size

        # COCO "images" entry
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

        # If there are stenosis annotations, parse them
        if "annotations" in item:
            stenosis_dict = item["annotations"]

            for ann_key, ann_val in stenosis_dict.items():
                # COCO uses XYWH format for bounding boxes
                if ann_key.startswith("bbox"):
                    xmin = float(ann_val["xmin"])
                    ymin = float(ann_val["ymin"])
                    xmax = float(ann_val["xmax"])
                    ymax = float(ann_val["ymax"])

                    w = xmax - xmin
                    h = ymax - ymin
                    area = w * h

                    # Skip invalid boxes
                    if w <= 0 or h <= 0:
                        continue

                    stenosis_annotations.append(
                        {
                            "id": stenosis_ann_id_counter,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [xmin, ymin, w, h],
                            "area": area,
                            "iscrowd": 0,
                        }
                    )
                    stenosis_ann_id_counter += 1

                elif ann_key.startswith("segmentation"):
                    seg_points = ann_val["xyxy"]
                    area = calculate_polygon_area(seg_points)
                    bbox = calculate_bbox_from_segmentation(seg_points)

                    stenosis_annotations.append(
                        {
                            "id": stenosis_ann_id_counter,
                            "image_id": image_id,
                            "category_id": 1,
                            "segmentation": [seg_points],  # list of polygons
                            "area": area,
                            "iscrowd": 0,
                            "bbox": bbox,  # [xmin, ymin, w, h]
                        }
                    )
                    stenosis_ann_id_counter += 1

    # Build final COCO JSON structure
    stenosis_coco_dict = {
        "images": stenosis_images,
        "annotations": stenosis_annotations,
        "categories": stenosis_categories,
    }

    # Save the file
    stenosis_json_path = datasets_path / "coco_detection.json"
    with open(stenosis_json_path, "w") as f_stenosis:
        json.dump(stenosis_coco_dict, f_stenosis, indent=2)

    print(f"[COCO Detection] Stenosis COCO JSON saved to: {stenosis_json_path}")
    return str(stenosis_json_path)


###########################
# Segmentation (Arteries)
###########################
def construct_coco_segmentation(
    json_path: Union[str, Path], root_folder: Union[str, Path]
) -> str:
    """
    Build a COCO-style annotation for the "Arteries_Segmentation" portion
    of your dataset and save it as 'arteries_coco.json' in:
        <root_folder>/datasets/arteries/

    Returns:
        str: The path to the newly created 'arteries_coco.json' file.
    """
    root_path = Path(root_folder).resolve()
    datasets_path = root_path / "datasets" / "coco"
    datasets_path.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    if "Arteries_Segmentation" not in data:
        raise KeyError(
            "JSON does not contain 'Arteries_Segmentation' at its top level."
        )

    # Data subset for segmentation
    arteries_data = data["Arteries_Segmentation"]

    # Prepare COCO-style lists
    arteries_images = []
    arteries_annotations = []
    arteries_categories = [{"id": 1, "name": "artery"}]  # Single class

    arteries_ann_id_counter = 1
    image_id_counter = 1

    # Iterate over each item in 'Arteries_Segmentation'
    for key, item in arteries_data.items():
        img_path = item["image"]["dataset_route"]

        # Use PIL to read image dimensions
        with Image.open(img_path) as im:
            width, height = im.size

        # COCO "images" entry
        image_id = image_id_counter
        image_id_counter += 1
        arteries_images.append(
            {
                "id": image_id,
                "file_name": os.path.basename(img_path),
                "width": width,
                "height": height,
            }
        )

        # If there are vessel segmentations, parse them
        if "annotations" in item and "vessel_segmentations" in item["annotations"]:
            vessel_segments = item["annotations"]["vessel_segmentations"]

            for seg_info in vessel_segments:
                # Each seg_info may contain "segment0", "segment1", or arbitrary keys
                # that contain polygon points. Some also contain bounding boxes, etc.
                # Below is a simpler approach: we gather all polygon coordinates
                # in a single segmentation annotation.

                # If you have multiple polygons, you can create multiple annotations
                # or group them in a list, depending on your labeling structure.
                for seg_key, seg_val in seg_info.items():
                    if seg_key.startswith("segment"):
                        seg_points = seg_val  # flat [x1,y1,x2,y2,...]
                        area = calculate_polygon_area(seg_points)
                        bbox = calculate_bbox_from_segmentation(seg_points)

                        arteries_annotations.append(
                            {
                                "id": arteries_ann_id_counter,
                                "image_id": image_id,
                                "category_id": 1,
                                "segmentation": [seg_points],
                                "area": area,
                                "iscrowd": 0,
                                "bbox": bbox,
                            }
                        )
                        arteries_ann_id_counter += 1

    # Build final COCO JSON structure
    arteries_coco_dict = {
        "images": arteries_images,
        "annotations": arteries_annotations,
        "categories": arteries_categories,
    }

    # Save the file
    arteries_json_path = datasets_path / "coco_segmentation.json"
    with open(arteries_json_path, "w") as f_arteries:
        json.dump(arteries_coco_dict, f_arteries, indent=2)

    print(f"[COCO Segmentation] Arteries COCO JSON saved to: {arteries_json_path}")
    return str(arteries_json_path)
