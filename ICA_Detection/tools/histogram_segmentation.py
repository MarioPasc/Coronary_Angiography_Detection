# ica_yolo_detection/preprocessing/tools/histogram_segmentation.py

import cv2
import numpy as np
from typing import Any

def segment_arteries_adaptive(image: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
    """
    Returns a binary mask using OpenCV's Adaptive Thresholding (Gaussian method).
    This method computes a local threshold for each block (block_size x block_size) and 
    subtracts a constant C from the weighted sum of neighborhood values.
    
    Args:
        image (np.ndarray): 2D grayscale image (e.g., coronary angiogram).
        block_size (int): Size of the neighborhood used to calculate the threshold (must be odd).
        C (int): Constant subtracted from the computed mean.
    
    Returns:
        np.ndarray: Binary mask where pixels above the local threshold are set to 255.
    """
    mask = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )
    return mask

if __name__ == "__main__":
    import cv2
    input_img = "path/to/gray_image.png"
    img = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        mask = segment_arteries_adaptive(img, block_size=11, C=2)
        cv2.imwrite("adaptive_mask.png", mask)
        print("Adaptive segmentation completed.")
