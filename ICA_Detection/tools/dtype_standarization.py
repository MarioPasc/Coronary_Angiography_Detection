# ica_yolo_detection/preprocessing/tools/dtype_standarization.py

import cv2
import numpy as np
from typing import Optional, Dict

def apply_dtype_standarization(image_path: str, output_path: str, desired_dtype: str) -> Optional[Dict[str, str]]:
    """
    Convert the image at image_path to the desired data type if it differs.
    Reads the image using OpenCV and checks its dtype. If conversion is needed,
    the image is converted using NumPy's astype() and saved to output_path.
    
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the converted image.
        desired_dtype (str): Desired NumPy data type as a string (e.g., "uint8", "float32").
    
    Returns:
        Optional[Dict[str, str]]: A dictionary with keys "original" and "final" indicating the dtypes,
                                  or None if the image could not be read.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    original_dtype = str(img.dtype)
    if original_dtype == desired_dtype:
        cv2.imwrite(output_path, img)
        return {"original": original_dtype, "final": desired_dtype}
    
    try:
        desired_np_dtype = np.dtype(desired_dtype)
        converted_img = img.astype(desired_np_dtype)
        cv2.imwrite(output_path, converted_img)
        return {"original": original_dtype, "final": desired_dtype}
    except Exception as e:
        print(f"Error converting image dtype for {image_path}: {e}")
        return None

if __name__ == "__main__":
    # Example usage:
    input_img = "path/to/input.bmp"  # Could also be a .png
    output_img = "path/to/converted.bmp"
    result = apply_dtype_standarization(input_img, output_img, desired_dtype="uint8")
    if result is not None:
        print(f"Image dtype conversion: {result}")
