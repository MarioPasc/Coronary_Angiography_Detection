# ica_yolo_detection/preprocessing/tools/format_standarization.py

import cv2
from typing import Optional

def apply_format_standarization(image_path: str, output_path: str, desired_format: str) -> Optional[any]:
    """
    Convert the image at image_path to the desired file format using OpenCV.
    
    The function reads the image and writes it to output_path. The desired_format (e.g., "png", "bmp")
    is expected to match the extension of output_path.
    
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the converted image.
        desired_format (str): Desired image format (without dot, e.g., "png").
    
    Returns:
        Optional: The image as a NumPy array after conversion if successful; otherwise, None.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    # Write the image using OpenCV; output_path's extension determines the format.
    success = cv2.imwrite(output_path, img)
    if not success:
        print(f"Error: Could not write image to {output_path}")
        return None
    return img

if __name__ == "__main__":
    input_img = "path/to/input.bmp"  # e.g., a bmp image.
    output_img = "path/to/converted.png"  # Desired output format: png.
    result = apply_format_standarization(input_img, output_img, desired_format="png")
    if result is not None:
        print("Image format conversion successful.")
