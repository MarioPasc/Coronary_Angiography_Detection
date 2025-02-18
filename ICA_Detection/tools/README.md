# Tools Module

The **Preprocessing Tools** module contains a collection of scripts designed to execute specific image processing tasks as defined by the preprocessing plan. These tasks are applied to standardized images before they are used in downstream tasks (e.g., training detection models).

The tools module is organized into several submodules, each responsible for a distinct transformation:

- **Resolution Standardization (`resolution.py`):**  
  Uses OpenCV to resize an image to a target resolution based on a user-specified interpolation method (e.g., "bilinear", "bicubic").

- **Dtype Standardization (`dtype_standarization.py`):**  
  Reads an image and converts its NumPy data type (dtype) to a desired type (e.g., "uint8") if necessary, using OpenCV and NumPy.

- **Low-Pass Filtering (`lowpass.py`):**  
  Implements Gaussian smoothing. It computes a Gaussian kernel given a window size and sigma, normalizes it, and applies OpenCV's `filter2D` to smooth the image.

- **Filtering Smoothing Equalization (`fse.py` – integrated into the FSE step):**  
  Applies the low-pass filtering step (via `lowpass.py`) followed by histogram equalization to enhance image contrast.

- **High-Pass Processing (`highpass.py`):**  
  Provides functions to detect edges using first-derivative (Sobel operator) and second-derivative (Laplacian operator) methods, producing binary edge maps.

- **Histogram Segmentation (`histogram_segmentation.py`):**  
  Uses adaptive thresholding (with OpenCV’s adaptive methods) to generate binary masks for artery segmentation in images with varying illumination.

- **Connected Components Analysis (`connected_components.py`):**  
  Contains functions to extract the largest connected component, assign random colors to each component for visualization, and filter connected components by area.

- **Format Standardization (`format_standarization.py`):**  
  Converts the image to a desired file format (e.g., converting BMP to PNG) by simply re-saving the image with the correct extension using OpenCV.

## Detailed Descriptions

### Resolution Standardization
- **Function:** `apply_resolution(image_path, output_path, x_final, y_final, method)`
- **Description:** Resizes the image using OpenCV's `cv2.resize` and the specified interpolation method.

### Dtype Standardization
- **Function:** `apply_dtype_standarization(image_path, output_path, desired_dtype)`
- **Description:** Reads the image, checks its dtype, converts it if necessary, and saves the result.

### Low-Pass Filtering
- **Function:** `apply_lowpass(image, window_size, sigma)`
- **Description:** Computes a Gaussian kernel and applies it to the image using OpenCV’s `filter2D` for smoothing.

### Filtering Smoothing Equalization (FSE)
- **Usage:** Calls `apply_lowpass` to smooth the image and then performs histogram equalization:
  - Histogram is computed, the cumulative distribution function (CDF) is normalized, and the pixel values are remapped accordingly.

### High-Pass Processing
- **Functions:** 
  - `edge_detection_first_derivative(mask)` – Uses Sobel operators.
  - `edge_detection_laplacian(mask)` – Uses the Laplacian operator.
- **Description:** Both functions produce binary edge maps.

### Histogram Segmentation
- **Function:** `segment_arteries_adaptive(image, block_size, C)`
- **Description:** Applies adaptive thresholding (Gaussian or Mean) to segment arteries from the background.

### Connected Components Analysis
- **Functions:**
  - `largest_connected_component(mask)` – Extracts the largest connected component.
  - `color_connected_components(mask)` – Returns a color image with random colors for each component.
  - `filter_connected_components_by_area(mask, min_pixels)` – Filters components by area.
- **Description:** Useful for isolating relevant anatomical structures from binary masks.

### Format Standardization
- **Function:** `apply_format_standarization(image_path, output_path, desired_format)`
- **Description:** Converts an image to the desired file format by re-saving it.

