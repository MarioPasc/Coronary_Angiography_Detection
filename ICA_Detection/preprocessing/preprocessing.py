# ica_yolo_detection/preprocessing/preprocessing.py

import os
import cv2
import json
import shutil
from typing import List
from tqdm import tqdm  # type: ignore

# Import our tool functions.

from ICA_Detection.tools.format_standarization import apply_format_standarization
from ICA_Detection.tools.dtype_standarization import apply_dtype_standarization
from ICA_Detection.tools.resolution import apply_resolution
from ICA_Detection.tools.fse import filtering_smoothing_equalization


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
    # Create output structure.
    images_out = os.path.join(out_dir, "images")
    labels_out = os.path.join(out_dir, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    # Load the planned JSON.
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

        # Define working filename: we use PNG for all processed images.
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

            # Retrieve original dimensions before resizing.
            orig_width = img_info.get("width")
            orig_height = img_info.get("height")

            new_img_path = current_img_path  # Overwrite in place.
            ret = apply_resolution(
                current_img_path, new_img_path, desired_X, desired_Y, method
            )
            if ret is None:
                print(f"Error resizing image for {uid}.")
                continue
            # Update image dimensions in JSON.
            img_info["width"] = desired_X
            img_info["height"] = desired_Y

        # --- Step 4: Filtering Smoothing Equalization ---
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
        # Save label file only for images with lesion=True.
        if entry.get("lesion", False):
            annotations = entry.get("annotations", {})
            lines = []
            # For each bounding box, write a line in YOLO format:
            # "stenosis x_center y_center width height"
            for key, bbox in annotations.items():
                if key == "name":
                    continue
                line = f"0 {bbox.get('x_center')} {bbox.get('y_center')} {bbox.get('width')} {bbox.get('height')}"
                lines.append(line)
            label_filename = annotations.get("name", f"{uid}.txt")
            label_out_path = os.path.join(labels_out, label_filename)
            with open(label_out_path, "w") as f:
                f.write("\n".join(lines))


if __name__ == "__main__":

    plan_steps = {
        "format_standarization": {"desired_format": "png"},
        "dtype_standarization": {"desired_dtype": "uint8"},
        "resolution_standarization": {
            "desired_X": 512,
            "desired_Y": 512,
            "method": "bilinear",
        },
        # "filtering_smoothing_equalization": {"window_size": 5, "sigma": 1.0}
    }

    output_base_folder = "/home/mariopasc/Python/Datasets/COMBINED"
    output_planned_json = os.path.join(output_base_folder, "planned_standardized.json")
    steps_order = list(plan_steps.keys())

    output_base_folder = os.path.join(output_base_folder, "ICA_DETECTION")
    process_images(output_planned_json, output_base_folder, steps_order)
    print("Preprocessing completed.")
