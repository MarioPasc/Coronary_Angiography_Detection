# ica_yolo_detection/preprocessing/tools/highpass.py

import cv2
import numpy as np
from typing import Any

def edge_detection_sobel(mask: np.ndarray) -> np.ndarray:
    """
    Applies a first-derivative (Sobel) operator to the input image and returns a binary edge map.
    
    Args:
        mask (np.ndarray): Input image (e.g., a grayscale image) as a NumPy array.
    
    Returns:
        np.ndarray: Binary edge map (uint8) obtained by thresholding the gradient magnitude.
    """
    sobel_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x**2 + sobel_y**2)
    mag_8u = np.uint8(np.clip(mag, 0, 255))
    _, edges = cv2.threshold(mag_8u, 50, 255, cv2.THRESH_BINARY)
    return edges

def edge_detection_laplacian(mask: np.ndarray) -> np.ndarray:
    """
    Applies a Laplacian (second-derivative) operator to the input image and returns a binary edge map.
    
    Args:
        mask (np.ndarray): Input image (e.g., a grayscale image) as a NumPy array.
    
    Returns:
        np.ndarray: Binary edge map (uint8) obtained by thresholding the absolute Laplacian.
    """
    lap = cv2.Laplacian(mask, cv2.CV_64F, ksize=3)
    lap_8u = np.uint8(np.clip(np.abs(lap), 0, 255))
    _, edges = cv2.threshold(lap_8u, 50, 255, cv2.THRESH_BINARY)
    return edges

if __name__ == "__main__":
    # Example usage:
    import cv2
    input_img = "path/to/gray_image.png"
    img = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        edges_sobel = edge_detection_sobel(img)
        edges_laplacian = edge_detection_laplacian(img)
        cv2.imwrite("sobel_edges.png", edges_sobel)
        cv2.imwrite("laplacian_edges.png", edges_laplacian)
        print("Edge detection completed.")
