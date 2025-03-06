import os
import shutil
from pathlib import Path
from typing import Union, Any, Dict
import json
from PIL import Image


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


def construct_pytorch_compatible(
    json_path: Union[str, Path], root_folder: Union[str, Path], dataset_name: str
) -> str:
    """
    Creates a dataset folder (if needed) and two JSON files:
      1) The original "Standard_dataset" JSON, with possibly updated paths
      2) A COCO-style JSON, to enable COCO evaluation with CocoEvaluator.

    Returns the path to the newly created  COCO JSON.
    """

    root_path = Path(root_folder).resolve()
    datasets_path = root_path / "datasets"
    datasets_path.mkdir(parents=True, exist_ok=True)

    dataset_name = dataset_name.strip("_")
    output_dir = datasets_path / dataset_name
    os.makedirs(output_dir, exist_ok=True)

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
    # 2) Create a COCO-style JSON for evaluation
    #
    #    "images":      [ { "id", "file_name", "width", "height" }, ... ]
    #    "annotations": [ { "id", "image_id", "category_id", "bbox", "area", "iscrowd" }, ... ]
    #    "categories":  [ { "id", "name" }, ... ]
    #
    #    We'll assume there's only 1 class: "lesion" => category_id=1
    #    If you have more classes, you'll need to adapt accordingly.
    # -----------------------------------------------------------------
    coco_images = []
    coco_annotations = []
    coco_categories = [{"id": 1, "name": "stenosis"}]  # adapt if more classes

    ann_id_counter = 1  # unique ID for each annotation
    image_id_counter = 1

    for key, item in new_data["Standard_dataset"].items():
        img_path = item["image"]["dataset_route"]

        # 2.1) Load the image to get width/height
        #      You can skip this if you already have width/height in your data
        #      But COCO format requires them.
        with Image.open(img_path) as im:
            width, height = im.size

        # 2.2) Add an entry to the "images" list
        #      We'll map your original `key` or integer index to a new integer ID
        image_id = image_id_counter
        image_id_counter += 1

        coco_images.append(
            {
                "id": image_id,
                "file_name": os.path.basename(
                    img_path
                ),  # or the full path if you prefer
                "width": width,
                "height": height,
            }
        )

        # 2.3) Convert bounding boxes from item["annotations"]
        annots = item.get("annotations", {})
        for ann_key, ann_val in annots.items():
            if ann_key.startswith("bbox"):
                xmin = float(ann_val["xmin"])
                ymin = float(ann_val["ymin"])
                xmax = float(ann_val["xmax"])
                ymax = float(ann_val["ymax"])

                w = xmax - xmin
                h = ymax - ymin
                area = w * h
                if w <= 0 or h <= 0:
                    # skip invalid boxes
                    continue

                # Each annotation has a unique ID
                coco_annotations.append(
                    {
                        "id": ann_id_counter,
                        "image_id": image_id,
                        "category_id": 1,  # 'lesion'
                        "bbox": [xmin, ymin, w, h],  # XYWH format for COCO
                        "area": area,
                        "iscrowd": 0,
                    }
                )
                ann_id_counter += 1

    # 2.4) Build final COCO dict
    coco_dict = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }

    # 2.5) Save as dataset_name_coco.json
    coco_json_path = os.path.join(output_dir, f"{dataset_name}_coco.json")
    with open(coco_json_path, "w") as f_coco:
        json.dump(coco_dict, f_coco, indent=2)

    print(f"COCO-style annotation saved to: {coco_json_path}")

    return coco_json_path
