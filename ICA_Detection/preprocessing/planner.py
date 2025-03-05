# ica_yolo_detection/preprocessing/planner.py

import os
import cv2
import json
from typing import Dict, Any


def create_preprocessing_plan(
    data: Dict[str, Any], plan_steps: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Iterate through the standardized JSON dataset and add preprocessing instructions for each image
    based on the provided plan steps. The plan_steps dictionary may include keys such as:

    {
      "resolution_standarization": {"desired_X": int, "desired_Y": int, "method": str},
      "dtype_standarization": {"desired_dtype": str},
      "format_standarization": {"desired_format": str},
      "filtering_smoothing_equalization": {"window_size": int, "sigma": float}
    }

    For each image:
      - If its resolution (width/height) does not match the desired values, add a
        "resolution_standarization" tag.
      - If its data type does not match the desired dtype (determined by reading the image from its route),
        add a "dtype_standarization" tag.
      - If its file format (derived from "original_name") does not match the desired format,
        add a "format_standarization" tag.
      - Finally, unconditionally add the "filtering_smoothing_equalization" tag with the provided parameters.

    Args:
        data (Dict[str, Any]): The standardized JSON dataset.
        plan_steps (Dict[str, Any]): The desired preprocessing steps and parameters.

    Returns:
        Dict[str, Any]: The updated dataset with a new "preprocessing_plan" field added to each entry where needed.
    """
    dataset = data.get("Standard_dataset", {})
    for uid, entry in dataset.items():
        plan: Dict[str, Any] = entry.get("preprocessing_plan", {})
        img_info = entry.get("image", {})

        # Format check.
        format_plan = plan_steps.get("format_standarization")
        if format_plan:
            desired_format = format_plan.get("desired_format").lower()
            original_name = img_info.get("original_name", "")
            _, ext = os.path.splitext(original_name)
            original_format = ext.lstrip(".").lower()
            if original_format != desired_format:
                plan["format_standarization"] = {
                    "original_format": original_format,
                    "desired_format": desired_format,
                }

        # Dtype check.
        dtype_plan = plan_steps.get("dtype_standarization")
        if dtype_plan:
            desired_dtype = dtype_plan.get("desired_dtype")
            image_path = img_info.get("route")
            if image_path and os.path.exists(image_path):
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    original_dtype = str(img.dtype)
                    if original_dtype != desired_dtype:
                        plan["dtype_standarization"] = {
                            "original": original_dtype,
                            "desired": desired_dtype,
                        }
                else:
                    print(
                        f"Warning: Unable to read image at {image_path} for dtype check."
                    )
            else:
                print(f"Warning: Image path {image_path} does not exist.")

        # Resolution check.
        res_plan = plan_steps.get("resolution_standarization")
        if res_plan:
            desired_X = res_plan.get("desired_X")
            desired_Y = res_plan.get("desired_Y")
            method = res_plan.get("method")
            current_X = img_info.get("width")
            current_Y = img_info.get("height")
            if current_X != desired_X or current_Y != desired_Y:
                plan["resolution_standarization"] = {
                    "desired_X": desired_X,
                    "desired_Y": desired_Y,
                    "method": method,
                }

        # CLAHE methodology
        fse_plan = plan_steps.get("clahe")
        if fse_plan:
            plan["clahe"] = {
                "window_size": fse_plan.get("window_size"),
                "sigma": fse_plan.get("sigma"),
                "clipLimit": fse_plan.get("clipLimit"),
                "tileGridSize": fse_plan.get("tileGridSize")
            }

        # Filtering Smoothing Equalization is applied to all images if specified.
        fse_plan = plan_steps.get("filtering_smoothing_equalization")
        if fse_plan:
            plan["filtering_smoothing_equalization"] = {
                "window_size": fse_plan.get("window_size"),
                "sigma": fse_plan.get("sigma"),
            }

        # Ensure a labels_formats entry exists.
        if "dataset_formats" in plan_steps:
            plan["dataset_formats"] = plan_steps["dataset_formats"]
        else:
            # Default: produce YOLO labels.
            plan["dataset_formats"] = {"YOLO": True}

        if plan:
            entry["preprocessing_plan"] = plan
    return data


if __name__ == "__main__":
    # Example usage:
    input_json = "combined_standardized.json"
    output_json = "planned_standardized.json"

    with open(input_json, "r") as f:
        data = json.load(f)

    # Define all preprocessing steps in a single plan.
    plan_steps = {
        "resolution_standarization": {
            "desired_X": 512,
            "desired_Y": 512,
            "method": "bilinear",
        },
        "dtype_standarization": {"desired_dtype": "uint8"},
        "format_standarization": {"desired_format": "png"},
        "clahe":  {"window_size": 5, "sigma": 1.0, "clipLimit": 2.0, "tileGridSize": (8,8)},
        "filtering_smoothing_equalization": {"window_size": 5, "sigma": 1.0},
        "dataset_formats": {"YOLO": True},  # New key for additional label generation.
    }
    planned_data = create_preprocessing_plan(data, plan_steps)

    with open(output_json, "w") as f:
        json.dump(planned_data, f, indent=4)
    print(f"Preprocessing plan saved to {output_json}")
