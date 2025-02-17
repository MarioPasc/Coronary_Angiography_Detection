# ica_yolo_detection/preprocessing/planner.py

import os
import cv2
import json
from typing import Dict, Any

def create_preprocessing_plan(data: Dict[str, Any],
                              desired_width: int,
                              desired_height: int,
                              interp_method: str,
                              desired_dtype: str) -> Dict[str, Any]:
    """
    Iterate through the standardized JSON dataset and add preprocessing instructions for each image
    that does not match the desired resolution or desired data type.
    
    For resolution mismatches, a tag "resolution" is added with keys:
        - x_final: desired width,
        - y_final: desired height,
        - method: interpolation method.
    
    For data type mismatches, a tag "dtype_standarization" is added with keys:
        - original: the original image dtype,
        - final: the desired dtype.
    
    The function uses the image's "route" to load the image via OpenCV.
    
    Args:
        data (Dict[str, Any]): The standardized JSON dataset.
        desired_width (int): The desired final width.
        desired_height (int): The desired final height.
        interp_method (str): Interpolation method to be used (e.g., "bilinear", "nearest", etc.).
        desired_dtype (str): The desired numpy data type as a string (e.g., "uint8").
    
    Returns:
        Dict[str, Any]: The updated dataset with a new "preprocessing_plan" field in each entry that requires processing.
    """
    dataset = data.get("Standard_dataset", {})
    for uid, entry in dataset.items():
        plan = entry.get("preprocessing_plan", {})
        # Resolution planning: check if current resolution differs from desired.
        img_info = entry.get("image", {})
        current_width = img_info.get("width")
        current_height = img_info.get("height")
        if current_width != desired_width or current_height != desired_height:
            plan["resolution"] = {
                "x_final": desired_width,
                "y_final": desired_height,
                "method": interp_method
            }
        # Dtype planning: use cv2 to read the image from the "route".
        image_path = img_info.get("route")
        if image_path and os.path.exists(image_path):
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                original_dtype = str(img.dtype)
                if original_dtype != desired_dtype:
                    plan["dtype_standarization"] = {
                        "original": original_dtype,
                        "final": desired_dtype
                    }
            else:
                print(f"Warning: Unable to read image at {image_path} for dtype check.")
        else:
            print(f"Warning: Image path {image_path} does not exist.")
        # Update the entry's preprocessing plan if any changes were made.
        if plan:
            entry["preprocessing_plan"] = plan
    return data

if __name__ == "__main__":
    # Example usage:
    input_json = "combined_standardized.json"
    output_json = "planned_standardized.json"
    
    with open(input_json, "r") as f:
        data = json.load(f)
    
    # Set desired resolution to 1000x1000 (using bilinear interpolation) and dtype to "uint8"
    planned_data = create_preprocessing_plan(data, desired_width=1000, desired_height=1000,
                                             interp_method="bilinear", desired_dtype="uint8")
    
    with open(output_json, "w") as f:
        json.dump(planned_data, f, indent=4)
    print(f"Preprocessing plan saved to {output_json}")

"""
Great! let's add two more steps. Remember to give me the code for the step-specific .py file at tools/ and the updated planner code. Let's change the planner to recieve a JSON with the steps that will check, for example:

{
resolution: {desired_X, desired_Y, method}
dtype_standarization: {desired_dtype}
}


"""