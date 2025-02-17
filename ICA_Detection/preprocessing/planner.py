# ica_yolo_detection/preprocessing/planner.py

import os
import cv2
import json
from typing import Dict, Any

def create_preprocessing_plan(data: Dict[str, Any], plan_steps: Dict[str, Any]) -> Dict[str, Any]:
    """
    Iterate through the standardized JSON dataset and add preprocessing instructions for each image
    based on the provided plan steps. The plan_steps dictionary may include:
    
    {
      "resolution_standarization": {"desired_X": int, "desired_Y": int, "method": str},
      "dtype_standarization": {"desired_dtype": str},
      "format_standarization": {"desired_format": str}
    }
    
    For each image:
      - If its resolution (width/height) does not match desired_X/desired_Y, add a 
        "resolution_standarization" tag.
      - If its data type does not match desired_dtype (determined by reading the image from the route),
        add a "dtype_standarization" tag.
      - If its file format (derived from "original_name") does not match the desired_format,
        add a "format_standarization" tag.
    
    Args:
        data (Dict[str, Any]): The standardized JSON dataset.
        plan_steps (Dict[str, Any]): The desired preprocessing steps and parameters.
    
    Returns:
        Dict[str, Any]: The updated dataset with a "preprocessing_plan" field added where necessary.
    """
    dataset = data.get("Standard_dataset", {})
    for uid, entry in dataset.items():
        plan: Dict[str, Any] = entry.get("preprocessing_plan", {})
        img_info = entry.get("image", {})
        # --- Resolution check ---
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
                    "method": method
                }
        # --- Dtype check ---
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
                            "desired": desired_dtype
                        }
                else:
                    print(f"Warning: Unable to read image at {image_path} for dtype check.")
            else:
                print(f"Warning: Image path {image_path} does not exist.")
        # --- Format check ---
        format_plan = plan_steps.get("format_standarization")
        if format_plan:
            desired_format = format_plan.get("desired_format").lower()
            original_name = img_info.get("original_name", "")
            _, ext = os.path.splitext(original_name)
            original_format = ext.lstrip(".").lower()
            if original_format != desired_format:
                plan["format_standarization"] = {
                    "original_format": original_format,
                    "desired_format": desired_format
                }
        if plan:
            entry["preprocessing_plan"] = plan
    return data

if __name__ == "__main__":
    # Example usage:
    input_json = "combined_standardized.json"
    output_json = "planned_standardized.json"
    
    with open(input_json, "r") as f:
        data = json.load(f)
    
    plan_steps = {
        "resolution_standarization": {"desired_X": 1000, "desired_Y": 1000, "method": "bilinear"},
        "dtype_standarization": {"desired_dtype": "uint8"},
        "format_standarization": {"desired_format": "png"}
    }
    
    planned_data = create_preprocessing_plan(data, plan_steps)
    
    with open(output_json, "w") as f:
        json.dump(planned_data, f, indent=4)
    print(f"Preprocessing plan saved to {output_json}")
