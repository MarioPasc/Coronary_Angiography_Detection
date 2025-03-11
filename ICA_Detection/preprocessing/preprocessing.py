# ica_yolo_detection/preprocessing/preprocessing.py

import os
import cv2
import json
import shutil
from typing import List, Dict, Any
from tqdm import tqdm  # type: ignore
from pathlib import Path
import numpy as np

from PIL import Image, ImageDraw

# Import our tool functions.

from ICA_Detection.tools.format_standarization import apply_format_standarization
from ICA_Detection.tools.dtype_standarization import apply_dtype_standarization
from ICA_Detection.tools.resolution import apply_resolution
from ICA_Detection.tools.fse import filtering_smoothing_equalization
from ICA_Detection.tools.clahe import clahe_enhancement
from ICA_Detection.tools.bbox_translation import common_to_yolo, rescale_bbox
from ICA_Detection.tools.dataset_conversions import (
    construct_yolo,
    construct_pytorch_compatible,
)


def process_images(json_path: str, out_dir: str, steps_order: List[str]) -> None:
    """
    Apply the preprocessing plan to all images based on the provided JSON and step order.

    This function creates the following folder structure under out_dir:

        out_dir/
            images/
            labels/

    It processes each image entry in the JSON according to the provided step order:

      1. format_standarization: If the image is not already in PNG, convert it using
         apply_format_standarization; otherwise, copy it.
      2. dtype_standarization: Standardize the image data type if needed.
      3. resolution_standarization: Resize the image if needed.
      4. filtering_smoothing_equalization: Apply Gaussian smoothing followed by histogram equalization.

    After processing, the final image is saved to out_dir/images.
    For images with "lesion": true, the annotations are saved as a text file in YOLO format
    (one bounding box per row) in out_dir/labels.

    Args:
        json_path (str): Path to the planned standardized JSON file.
        out_dir (str): Output directory for the final folder structure.
        steps_order (List[str]): Comma-separated list of preprocessing steps in order.
                                  Default order is:
                                  ["format_standarization", "dtype_standarization", "resolution_standarization", "filtering_smoothing_equalization"]
    """
    # Create output structure
    images_out = os.path.join(out_dir, "images")
    labels_out = os.path.join(out_dir, "labels_pascal_voc")
    labels_yolo_out = os.path.join(out_dir, "labels_yolo")
    datasets_out_dir = os.path.join(out_dir, "datasets")
    output_masks_dir = os.path.join(datasets_out_dir, "segmentation/masks")

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    os.makedirs(labels_yolo_out, exist_ok=True)
    os.makedirs(datasets_out_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)


    # Load the planned JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    dataset = data.get("Standard_dataset", {})

    for uid, entry in tqdm(
        iterable=dataset.items(),
        desc="Processing JSON entries ...",
        colour="green",
        total=len(dataset.items()),
    ):
        img_info = entry.get("image", {})
        orig_img_path = img_info.get("route")
        if not orig_img_path or not os.path.exists(orig_img_path):
            print(f"Skipping {uid}: original image path not found.")
            continue

        # Define working filename: we use PNG for all processed images
        working_filename = f"{uid}.png"
        working_img_path = os.path.join(images_out, working_filename)


        # --- Step 1: Format Standardization ---
        if (
            "format_standarization" in entry.get("preprocessing_plan", {})
            and "format_standarization" in steps_order
        ):
            desired_format = (
                entry["preprocessing_plan"]["format_standarization"]
                .get("desired_format")
                .lower()
            )
            ret = apply_format_standarization(
                orig_img_path, working_img_path, desired_format
            )
            if ret is None:
                print(f"Error converting format for {uid}.")
                continue
        else:
            shutil.copy2(orig_img_path, working_img_path)

        # Rename 'route' to 'original_route' and introduce 'dataset_route'
        img_info["original_route"] = img_info.pop("route")
        img_info["dataset_route"] = working_img_path

        current_img_path = working_img_path

        # --- Step 2: Dtype Standardization ---
        if (
            "dtype_standarization" in entry.get("preprocessing_plan", {})
            and "dtype_standarization" in steps_order
        ):
            desired_dtype = entry["preprocessing_plan"]["dtype_standarization"].get(
                "desired"
            )
            new_img_path = current_img_path
            ret = apply_dtype_standarization(
                current_img_path, new_img_path, desired_dtype
            )
            if ret is None:
                print(f"Error converting dtype for {uid}.")
                continue

        # --- Step 3: Resolution Standardization ---
        if (
            "resolution_standarization" in entry.get("preprocessing_plan", {})
            and "resolution_standarization" in steps_order
        ):
            res_plan = entry["preprocessing_plan"]["resolution_standarization"]
            desired_X = res_plan.get("desired_X")
            desired_Y = res_plan.get("desired_Y")
            method = res_plan.get("method")

            
            new_img_path = current_img_path  # Overwrite in place
            ret = apply_resolution(
                current_img_path, new_img_path, desired_X, desired_Y, method
            )
            if ret is None:
                print(f"Error resizing image for {uid}.")
                continue
            # --- Update JSON with new image resolution and bounding boxes ---
            # Save old dimensions
            old_width = img_info.get("width")
            old_height = img_info.get("height")
            # Update image info with new resolution
            img_info["width"] = desired_X
            img_info["height"] = desired_Y

            # Update the bounding box coordinates in the JSON (keeping Pascal VOC format)
            annotations = entry.get("annotations", {})
            for key, bbox in annotations.items():
                if key == "name":
                    continue
                elif key.startswith("segmentation"):
                    continue
                # Save the vessel segmentation
                elif key == "vessel_segmentations":
                    vessel_segmentations = annotations.get("vessel_segmentations", [])

                    mask_filename = f"{uid}_seg.png"
                    mask_path = os.path.join(output_masks_dir, mask_filename)

                    # Create an empty mask image
                    mask = Image.new("L", (old_width, old_height), 0)
                    draw = ImageDraw.Draw(mask)

                    # Draw all vessel segmentations on the mask
                    for vessel_seg in vessel_segmentations:
                        xyxy = vessel_seg.get("xyxy", [])
                        attributes = vessel_seg.get("attributes", [])
                        attributes["mask_path"] = mask_path
                        # Convert flat list to points array
                        if xyxy and len(xyxy) >= 4:
                            points = np.array(xyxy).reshape(-1, 2)
                            # Convert to tuple list for PIL
                            points = [(x, y) for x, y in points]
                            # Draw polygon with fill color 255 (white)
                            draw.polygon(points, fill=255)

                    # Resize mask to standard size
                    mask = mask.resize((desired_X, desired_Y), Image.NEAREST)

                    # Save mask to file
                    mask.save(mask_path)

                elif key.startswith("bbox"):
                    # Save the bbox
                    updated_bbox = rescale_bbox(
                        bbox, old_width, old_height, desired_X, desired_Y
                    )
                    annotations[key] = updated_bbox



        # --- CLAHE ---
        if "clahe" in entry.get("clahe", {}) and "clahe" in steps_order:
            fse_plan = entry["preprocessing_plan"]["clahe"]
            window_size = fse_plan.get("window_size")
            sigma = fse_plan.get("sigma")
            clipLimit = fse_plan.get("clipLimit")
            tileGridSize = fse_plan.get("tileGridSize")
            img = cv2.imread(current_img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error reading image for FSE for {uid}.")
                continue
            enhanced = clahe_enhancement(
                img, window_size, sigma, clipLimit, tileGridSize
            )
            cv2.imwrite(current_img_path, enhanced)

        # --- Filtering Smoothing Equalization ---
        if (
            "filtering_smoothing_equalization" in entry.get("preprocessing_plan", {})
            and "filtering_smoothing_equalization" in steps_order
        ):
            fse_plan = entry["preprocessing_plan"]["filtering_smoothing_equalization"]
            window_size = fse_plan.get("window_size")
            sigma = fse_plan.get("sigma")
            img = cv2.imread(current_img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error reading image for FSE for {uid}.")
                continue
            enhanced = filtering_smoothing_equalization(img, window_size, sigma)
            cv2.imwrite(current_img_path, enhanced)

        # --- Save Annotations ---
        if entry.get("lesion", False):
            annotations = entry.get("annotations", {})
            label_filename = annotations.get("name", f"{uid}.txt")

            lines = []
            for key, bbox in annotations.items():
                if key == "name":
                    continue
                elif key.startswith("segmentation"):
                    continue
                elif key == "vessel_segmentations":
                    continue
                xmin = bbox["xmin"]
                ymin = bbox["ymin"]
                xmax = bbox["xmax"]
                ymax = bbox["ymax"]
                line = f"0 {xmin} {ymin} {xmax} {ymax}"
                lines.append(line)

            # Save default labels file
            label_out_path = os.path.join(labels_out, label_filename)
            with open(label_out_path, "w") as f:
                f.write("\n".join(lines))

            # --- Additional Label Formats ---
            labels_formats = entry.get("preprocessing_plan", {}).get(
                "dataset_formats", {}
            )
            if labels_formats.get("YOLO", False):
                label_yolo_out_path = os.path.join(labels_yolo_out, label_filename)
                lines = []
                for key, bbox in annotations.items():
                    if key == "name":
                        continue
                    elif key.startswith("segmentation"):
                        continue
                    elif key == "vessel_segmentations":
                        continue
                    orig_width = img_info.get("width")
                    orig_height = img_info.get("height")
                    yolo_bbox = common_to_yolo(bbox, orig_width, orig_height)
                    line = f"0 {yolo_bbox['x_center']} {yolo_bbox['y_center']} {yolo_bbox['width']} {yolo_bbox['height']}"
                    lines.append(line)

                with open(label_yolo_out_path, "w") as f:
                    f.write("\n".join(lines))

    # --------------------------------
    # After processing all, save JSON:
    # --------------------------------
    processed_json_path = os.path.join(out_dir, "processed.json")
    with open(processed_json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"All images processed. Updated JSON saved to: {processed_json_path}")

    # We only need one entry to check for the datasets, since all the images are going to
    # be in all the datasets, therefore the dataset_formats flag is present in all JSON entries
    config = entry.get("preprocessing_plan", {})
    generate_datasets(root_folder=out_dir, config=config, json_path=processed_json_path)


def generate_datasets(root_folder: str, config: Dict[str, Any], json_path: str) -> None:
    """
    Look at config['dataset_formats'] to determine which dataset
    formats should be generated. Then call the corresponding function
    from dataset_conversions.

    :param root_folder: Path to the root folder that contains:
                        images/, labels_pascal_voc/, labels_yolo/, processed.json, etc.
    :param config: A dictionary that has at least a 'dataset_formats' key
                   indicating which dataset formats to create:
                     {
                       "dataset_formats": {
                         "YOLO": True,
                         "RetinaNet": False,
                         "FasterRCNN": True,
                         ...
                       }
                     }
    """
    dataset_formats = config.get("dataset_formats", {})

    # Create a datasets/ subfolder if needed
    root_path = Path(root_folder).resolve()
    datasets_path = root_path / "datasets"
    datasets_path.mkdir(exist_ok=True)

    print(f"Building datasets: ")
    print(dataset_formats)

    # For each format that is True, call the corresponding constructor
    if dataset_formats.get("YOLO", False):
        construct_yolo(root_path)

        # Cleanup
        shutil.rmtree(os.path.join(root_path, "labels_yolo"))

    if (
        dataset_formats.get("RetinaNet", False)
        or dataset_formats.get("FasterRCNN", False)
        or dataset_formats.get("SSD", False)
    ):
        construct_pytorch_compatible(
            json_path=json_path, root_folder=root_path, dataset_name="detection"
        )


def main():

    plan_steps = {
        "format_standarization": {"desired_format": "png"},
        "dtype_standarization": {"desired_dtype": "uint8"},
        "resolution_standarization": {
            "desired_X": 512,
            "desired_Y": 512,
            "method": "bilinear",
        },
        "clahe": {
            "window_size": 5,
            "sigma": 1.0,
            "clipLimit": 2.0,
            "tileGridSize": (8, 8),
        },
        "filtering_smoothing_equalization": {"window_size": 5, "sigma": 1.0},
        "dataset_formats": {"YOLO": True},
    }

    output_base_folder = "/home/mariopasc/Python/Datasets/COMBINED"
    output_planned_json = os.path.join(output_base_folder, "planned_standardized.json")
    steps_order = list(plan_steps.keys())

    output_base_folder = os.path.join(output_base_folder, "ICA_DETECTION")
    process_images(output_planned_json, output_base_folder, steps_order)
    print("Preprocessing completed.")
