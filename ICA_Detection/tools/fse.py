# ica_yolo_detection/preprocessing/tools/fse.py

import cv2
import numpy as np
import math

def filtering_smoothing_equalization(image: np.ndarray, window_size: int, sigma: float) -> np.ndarray:
    """
    Apply Filtering Smoothing Equalization to a grayscale image.
    
    This function performs two sequential steps on a grayscale image:
    
    1. **Gaussian Smoothing:**  
       A Gaussian kernel of size (window_size x window_size) is computed using:
         w(i,j) = (1 / (2*pi*sigma^2)) * exp( -((i - c)^2 + (j - c)^2) / (2*sigma^2) )
       where c = window_size // 2. The kernel is normalized so that its sum is 1,
       and the image is convolved with this kernel using OpenCV's filter2D.
    
    2. **Histogram Equalization:**  
       The histogram of the smoothed image is computed over 256 gray levels.
       The cumulative distribution function (CDF) is used to map each pixel value a to:
         H(a) = round( (CDF(a) - CDF(min)) / (M*N - 1) * (L - 1) )
       where M*N is the total number of pixels and L=256.
       
    Args:
        image (np.ndarray): Input grayscale image as a NumPy array.
        window_size (int): Size of the Gaussian filter window (e.g., 5 for a 5x5 window).
        sigma (float): Standard deviation for the Gaussian kernel.
    
    Returns:
        np.ndarray: The enhanced image after Gaussian smoothing and histogram equalization.
    """
    # --- Step 1: Gaussian Smoothing ---
    c = window_size // 2
    # Create a Gaussian kernel.
    kernel = np.zeros((window_size, window_size), dtype=np.float32)
    for i in range(window_size):
        for j in range(window_size):
            kernel[i, j] = (1 / (2 * math.pi * sigma**2)) * math.exp(-(((i - c)**2 + (j - c)**2) / (2 * sigma**2)))
    kernel_sum = np.sum(kernel)
    if kernel_sum != 0:
        kernel /= kernel_sum

    smoothed = cv2.filter2D(image, -1, kernel)
    
    # --- Step 2: Histogram Equalization ---
    # Assume image pixel range is 0-255.
    hist, _ = np.histogram(smoothed.flatten(), bins=256, range=(0,256))
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    # Normalize the CDF: (cdf - cdf_min) / (total_pixels - cdf_min) * 255.
    cdf_min = cdf_masked.min()
    total_pixels = smoothed.size
    cdf_normalized = (cdf_masked - cdf_min) * 255 / (total_pixels - cdf_min)
    cdf_final = np.ma.filled(cdf_normalized, 0).astype('uint8')
    
    equalized = cdf_final[smoothed]
    return equalized

if __name__ == "__main__":
    # Example usage:
    input_img_path = "path/to/grayscale_input.png"
    output_img_path = "path/to/enhanced.png"
    # Read the input image as grayscale.
    image = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read image at {input_img_path}")
    else:
        enhanced_image = filtering_smoothing_equalization(image, window_size=5, sigma=1.0)
        cv2.imwrite(output_img_path, enhanced_image)
        print("Enhanced image saved.")
