import cv2
import numpy as np
from ICA_Detection.tools.lowpass import apply_lowpass

def clahe_enhancement(
    image: np.ndarray,
    window_size: int = 5,
    sigma: float = 1.0,
    clipLimit: float = 2.0,
    tileGridSize: tuple = (8, 8),
    border_type: int = cv2.BORDER_CONSTANT
) -> np.ndarray:
    """
    FSE-like enhancement + CLAHE:
      1) Gaussian smoothing (5x5 by default) using OpenCV's filter2D
      2) Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE)
    """

    # Apply lowpass Gaussian smoothing
    smoothed = apply_lowpass(
        image=image, 
        window_size=window_size, 
        sigma=sigma, 
        border_type=border_type
    )

    # Convert to 8-bit
    smoothed_8u = np.clip(smoothed, 0, 255).astype(np.uint8)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_result = clahe.apply(smoothed_8u)

    return clahe_result
