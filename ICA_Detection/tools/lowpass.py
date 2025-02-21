# ica_yolo_detection/preprocessing/tools/lowpass.py

import cv2
import numpy as np
import math
from typing import Any

def apply_lowpass(image: np.ndarray, window_size:int = 5, sigma: float = 1.0, border_type: int = cv2.BORDER_CONSTANT) -> np.ndarray:
    """
    Apply a Gaussian low-pass filter (smoothing) to the input image.
    
    Computes a 2D Gaussian kernel of size (window_size x window_size) using:
      w(i,j) = (1 / (2*pi*sigma^2)) * exp(-(((i - c)^2 + (j - c)^2) / (2*sigma^2)))
    where c = window_size // 2. The kernel is normalized so that its sum equals 1, and then the image is
    convolved with this kernel using OpenCV's filter2D.
    
    Args:
        image (np.ndarray): Input image (can be grayscale or multi-channel).
        window_size (int): The size of the Gaussian kernel (e.g., 5 for a 5x5 kernel).
        sigma (float): The standard deviation for the Gaussian kernel.
    
    Returns:
        np.ndarray: The smoothed image.
    """
    c = window_size // 2
    kernel = np.zeros((window_size, window_size), dtype=np.float32)
    for i in range(window_size):
        for j in range(window_size):
            kernel[i, j] = (1 / (2 * math.pi * sigma**2)) * math.exp(-(((i - c)**2 + (j - c)**2) / (2 * sigma**2)))
    kernel /= np.sum(kernel)
    smoothed = cv2.filter2D(image, -1, kernel, borderType=border_type)
    return smoothed

if __name__ == "__main__":
    # Example usage:
    input_path = "path/to/input.png"
    output_path = "path/to/smoothed.png"
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        smoothed_img = apply_lowpass(img, window_size=5, sigma=1.0)
        cv2.imwrite(output_path, smoothed_img)
        print("Lowpass filtering applied successfully.")
    else:
        print("Error reading the image.")
