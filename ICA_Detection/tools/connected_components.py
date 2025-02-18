# ica_yolo_detection/preprocessing/tools/connected_components.py

import cv2
import numpy as np
from typing import Any

def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Extracts the largest connected component from a binary mask.
    
    Args:
        mask (np.ndarray): Binary image (uint8) with foreground = 255 and background = 0.
    
    Returns:
        np.ndarray: Binary mask containing only the largest connected component.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask  # No foreground found.
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    lcc_mask = np.zeros_like(mask)
    lcc_mask[labels == largest_label] = 255
    return lcc_mask

def color_connected_components(mask: np.ndarray) -> np.ndarray:
    """
    Returns a color image where each connected component in the binary mask is assigned a random color.
    
    Args:
        mask (np.ndarray): Binary image (uint8) with foreground = 255.
    
    Returns:
        np.ndarray: Color (BGR) image visualizing each connected component.
    """
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background as black.
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label in range(num_labels):
        colored[labels == label] = colors[label]
    return colored

def filter_connected_components_by_area(mask: np.ndarray, min_pixels: int = -1) -> np.ndarray:
    """
    Filters the connected components in a binary mask, keeping only those components with an area 
    greater than or equal to a specified minimum. If min_pixels is -1, then only components with an area 
    of at least (1/15) of the largest component are retained.
    
    Args:
        mask (np.ndarray): Binary image (uint8) with foreground = 255.
        min_pixels (int): Minimum area (in pixels) for a component to be retained. Default is -1.
    
    Returns:
        np.ndarray: Binary mask containing only the filtered connected components.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask  # No foreground found.
    largest_area = np.max(stats[1:, cv2.CC_STAT_AREA])
    if min_pixels == -1:
        min_pixels = int((1/15) * largest_area)
    filtered_mask = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_pixels:
            filtered_mask[labels == label] = 255
    return filtered_mask

if __name__ == "__main__":
    input_mask = "path/to/binary_mask.png"
    mask = cv2.imread(input_mask, cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        lcc = largest_connected_component(mask)
        colored = color_connected_components(mask)
        filtered = filter_connected_components_by_area(mask, min_pixels=-1)
        cv2.imwrite("lcc_mask.png", lcc)
        cv2.imwrite("colored_components.png", colored)
        cv2.imwrite("filtered_mask.png", filtered)
        print("Connected components processing completed.")
