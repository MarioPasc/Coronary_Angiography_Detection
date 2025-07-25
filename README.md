# Coronariography Angiography Detection 

<p align="center">
  <img src="./assets/paper1.png" alt="paper_title" />
</p>

Our paper, "Hyperparameter Optimization of YOLO Models for Invasive Coronary Angiography Lesion Detection and Assessment" has been published in [Computers in Biology and Medicine](https://www.sciencedirect.com/journal/computers-in-biology-and-medicine). Full article is available using this [here](https://www.sciencedirect.com/science/article/pii/S0010482525010480). Computers in Biology and Medicine is a Q1 journal in the areas of "Computer Science, Interdisciplinary Applications" (26/175) and "Health Informatics" (7/107, Biology).

<p align="center">
  <img src="./assets/abstract.png" alt="abstract" />
</p>

This repository offers a tool to apply the Bayesian and Evolutionary optimization strategies to YOLO-based models using the `optimization` module. The Optimization Module provides a configurable framework for systematically searching for optimal hyperparameter sets to improve model performance. The module supports multiple HPO strategies and is designed for scalability and robust experiment management.

To use this module for HPO tuning of any YOLO-based model (YOLOv8, YOLOv9, YOLOv10, etc.) you may create a custom configuration file (see examples in `Coronary_Angiography_Detection/ICA_Detection/optimization/cfg/files/picasso/yaml/`) and run the following line from your conda environment-activated terminal:
```bash
run_optimization --config [config file path] --gpu-ids [gpu ids to use (e.g., 0,1,2)]
```

Key submodules and functionalities include:

-   **Configuration (`cfg`):**
    -   Manages HPO configurations via YAML files, parsed into `BHOConfig` and `HyperparameterConfig` dataclasses (`config.py`).
    -   Provides default training parameters (`defaults.py`) which can be overridden by the HPO process.
    -   Includes example SLURM scripts for distributed execution (`files/picasso/slurm_optimization.sh`).

-   **Engine (`engine`):**
    -   **Bayesian Hyperparameter Optimization (hpo.py):** Implements the `BayesianHyperparameterOptimizer` class, which leverages the Optuna framework (`ICA_Detection/optimization/engine/hpo.py`). This class manages Optuna studies, including:
        -   Support for various samplers (e.g., TPE, Random, GPSampler, QMCSampler, CMA-ES, etc.).
        -   Integration with model trainers to execute individual trials.
        -   Management of parallel trials across multiple GPUs.
        -   Result logging, checkpointing, and visualization.
    -   **Ultralytics Evolutionary Search (ultralytics_es.py):** Provides the `UltralyticsESTuner` class for performing evolutionary hyperparameter search using the `model.tune()` method from Ultralytics YOLO (`ICA_Detection/optimization/engine/ultralytics_es.py`). It supports:
        -   Both stock Ultralytics YOLOv8 and the DCA-YOLOv8 fork.
        -   Dynamic model mapping and environment setup to ensure compatibility.
    -   **Trainers (`trainers/`):** Contains specific model training logic. For instance, `UltralyticsTrainer` handles training for standard Ultralytics YOLO models. 

-   **Pipeline (`pipeline`):**
    -   **Orchestrator (orchestrator.py):** The `run_hpo` function (`ICA_Detection/optimization/pipeline/orchestrator.py`) serves as the main entry point for initiating HPO. It:
        -   Parses the configuration file.
        -   Selects and initializes the appropriate optimizer (`BayesianHyperparameterOptimizer` or `UltralyticsESTuner`) based on the configuration.
        -   Manages GPU resources and launches the optimization process.

## Other modules overview

The **ICA_Detection** project aims to develop a system for the detection of coronary artery disease through the analysis of angiographic images. This document outlines the modules that have been completed to date.

### 1. Integration Module

The Integration Module is responsible for standardizing, processing, and integrating multiple coronary artery disease (CAD) image datasets into a single, unified JSON file. This harmonization of annotations and image metadata from diverse sources (CADICA, ARCADE, KEMEROVO) creates a common format suitable for training detection models (e.g., YOLO) and further analysis.

Key functionalities include:
- Processing of **CADICA**, **ARCADE**, and **KEMEROVO** datasets.
- Conversion of dataset-specific annotations into a unified JSON structure.
- Transformation of bounding box coordinates to the YOLO normalized format (`x_center`, `y_center`, `width`, `height`).
- Construction of unique identifiers for each image/annotation pair, incorporating metadata such as patient, video, frame, and split.
- The main script `integrate.py` orchestrates the processing of specified datasets and merges their outputs.

The final JSON output contains a top-level key `"Standard_dataset"` with entries for every processed image, detailing `id`, `dataset_origin`, `lesion` status, image metadata (name, route, original name, resolution), and annotation data (name, bounding boxes with labels).

### 2. Preprocessing Module

The Preprocessing Module prepares image data for analysis and model training. It features a **Preprocessing Planner** (`planner.py`) that analyzes a standardized dataset (in JSON format) and generates a preprocessing plan for each image. This plan outlines the necessary transformations to meet desired criteria.

Key functionalities of the planner include:
- **Resolution Standardization:** Tags images for resizing if their dimensions do not match target criteria, specifying the interpolation method.
- **Data Type (Dtype) Standardization:** Tags images for dtype conversion (e.g., to `uint8`) if the current dtype differs from the target.
- **Format Standardization:** Tags images for file format conversion (e.g., BMP to PNG) if the current format is not the desired one.
- **Filtering, Smoothing, and Equalization (FSE):** Unconditionally tags images with parameters for an FSE step to enhance image quality.

The output of the planner is a JSON structure that guides the application of specific tools from the tools module.

### 3. Tools Module
*Path: tools*
*Documentation: README.md*

The Tools Module comprises a collection of scripts that execute specific image processing tasks as defined by the preprocessing plan generated by the Preprocessing Module. These tools operate on standardized images before they are utilized in downstream tasks, such as training detection models.

The module includes submodules for distinct transformations:
- **Resolution Standardization (`resolution.py`):** Resizes images to a target resolution using specified interpolation methods (e.g., bilinear, bicubic) via OpenCV.
- **Dtype Standardization (`dtype_standarization.py`):** Converts image data types to a desired type (e.g., `uint8`) using OpenCV and NumPy.
- **Low-Pass Filtering (`lowpass.py`):** Implements Gaussian smoothing by computing and applying a Gaussian kernel.
- **Filtering Smoothing Equalization (`fse.py`):** Integrates low-pass filtering with histogram equalization to enhance image contrast.
- **High-Pass Processing (`highpass.py`):** Provides edge detection functionalities using Sobel and Laplacian operators.
- **Histogram Segmentation (`histogram_segmentation.py`):** Applies adaptive thresholding for artery segmentation.
- **Connected Components Analysis (`connected_components.py`):** Includes functions for extracting, coloring, and filtering connected components by area.
- **Format Standardization (`format_standarization.py`):** Converts image file formats by re-saving images with the correct extension using OpenCV.

