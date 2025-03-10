# ica_yolo_detection/preprocessing/tools/resolution.py

import cv2
from typing import Optional, Any


def apply_resolution(
    image_path: str,
    output_path: str,
    x_final: int,
    y_final: int,
    method: str = "bilinear",
) -> Optional[Any]:
    """
    Resize the image at image_path to the desired resolution using OpenCV and save it to output_path.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the resized image.
        x_final (int): Desired width (in pixels).
        y_final (int): Desired height (in pixels).
        method (str): Interpolation method to use. Supported: "nearest", "bilinear", "bicubic", "lanczos". Defaults to "bilinear".

    Returns:
        Optional: The resized image as a NumPy array if successful; otherwise, None.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    interp_flag = interp_map.get(method.lower(), cv2.INTER_LINEAR)

    resized_img = cv2.resize(img, (x_final, y_final), interpolation=interp_flag)
    cv2.imwrite(output_path, resized_img)
    return resized_img


if __name__ == "__main__":
    # Example usage:
    input_img = "path/to/input.png"
    output_img = "path/to/resized.png"
    result = apply_resolution(input_img, output_img, 1000, 1000, method="bilinear")
    if result is not None:
        print("Image resized successfully.")
