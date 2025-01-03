#!/usr/bin/env python3
# coding: utf-8
"""
Automatic Inference Pipeline for YOLOv8 Models across 3 Folds,
including Ground Truth visualization as the first subplot column.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from typing import List, Dict, Any, Tuple
from typing_extensions import TypedDict
import numpy as np

try:
    from CADICA_Detection.external.ultralytics.ultralytics import YOLO
except ImportError as e:
    raise ImportError("Please install ultralytics >= 8.0.0") from e


# -------------------
# Logging Setup
# -------------------
logging.basicConfig(
    filename='automatic_inference_log.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# -------------------
# Constants
# -------------------
MODEL_NAMES = ["TPE", "GP-BHO", "Simulated_Annealing", "Baseline"]
FOLD_NAMES = ["fold_1", "fold_2", "fold_3"]
LESION_FOLDERS = ["p100", "p99", "p90_98", "p70_90", "p50_70", "nolesion"]

# Default YOLO inference params
INFERENCE_PARAMS: Dict[str, Any] = {
    'conf': 0.25,
    'iou': 0.5,
    'imgsz': 512,
    'half': False,
    'device': "cuda:0",
    'max_det': 300,
    'vid_stride': 1,
    'stream_buffer': False,
    'visualize': False,
    'augment': False,
    'agnostic_nms': False,
    'classes': None,
    'retina_masks': False,
    'embed': None,
    'project': None,
    'name': None,
    'show': False,
    'save': True,
    'save_frames': False,
    'save_txt': False,
    'save_conf': False,
    'save_crop': False,
    'show_labels': True,
    'show_conf': True,
    'show_boxes': True,
    'line_width': None,
}

# Visualization params for result.plot()
VISUALIZATION_PARAMS: Dict[str, Any] = {
    'conf': True,
    'line_width': 1,
    'font_size': 12,
    'font': 'Helvetica.ttf',
    'pil': False,
    'img': None,
    'im_gpu': None,
    'kpt_radius': 3,
    'kpt_line': True,
    'labels': True,
    'boxes': True,
    'probs': False,
    'show': False,
    'color_mode': 'class',
}


class WeightsPathMap(TypedDict):
    """
    TypedDict for storing paths to best.pt for each model in each fold.
    Example:
        {
            "fold_1": {
                "TPE":    "/path/to/TPE_outer_1_inner_1/weights/best.pt",
                "GP-BHO": "/path/to/GPSAMPLER_outer_1_inner_1/weights/best.pt",
                ...
            },
            "fold_2": { ... },
            "fold_3": { ... }
        }
    """
    TPE: str
    GP_BHO: str
    Simulated_Annealing: str
    Baseline: str


class FoldWeightsMap(TypedDict):
    fold_1: WeightsPathMap
    fold_2: WeightsPathMap
    fold_3: WeightsPathMap


def build_resources_dict(
    root_csv_folder: str,
    root_weights_folder: str
) -> Tuple[Dict[str, str], FoldWeightsMap]:
    """
    Build a dictionary for CSV paths and a dictionary for weights paths.
    Logs discovered paths for clarity.

    Args:
        root_csv_folder (str): Path containing 'fold_1', 'fold_2', 'fold_3' subfolders each with test.csv.
        root_weights_folder (str): Path containing model folders like TPE_outer_1_inner_1, etc.

    Returns:
        Tuple[Dict[str, str], FoldWeightsMap]:
            - A dict mapping fold name to the discovered test.csv path.
            - A dictionary of dictionaries mapping each fold to each model best.pt path.
    """
    fold_csv_paths: Dict[str, str] = {}
    weights_map: FoldWeightsMap = {
        "fold_1": {
            "TPE": "",
            "GP_BHO": "",
            "Simulated_Annealing": "",
            "Baseline": ""
        },
        "fold_2": {
            "TPE": "",
            "GP_BHO": "",
            "Simulated_Annealing": "",
            "Baseline": ""
        },
        "fold_3": {
            "TPE": "",
            "GP_BHO": "",
            "Simulated_Annealing": "",
            "Baseline": ""
        },
    }

    # Discover test.csv for each fold
    for fold in FOLD_NAMES:
        test_csv_path = os.path.join(root_csv_folder, fold, "test.csv")
        if os.path.isfile(test_csv_path):
            fold_csv_paths[fold] = test_csv_path
            logging.info(f"Detected {fold} test.csv => {test_csv_path}")
        else:
            logging.warning(f"Missing test.csv in {fold}: {test_csv_path}")

    # Discover best.pt for each fold and each model
    # e.g., TPE_outer_1_inner_1/weights/best.pt, etc.
    for fold_idx, fold in enumerate(FOLD_NAMES, start=1):
        # TPE
        tpe_dir = f"TPE_outer_{fold_idx}_inner_1"
        tpe_path = os.path.join(root_weights_folder, tpe_dir, "weights", "best.pt")
        if os.path.isfile(tpe_path):
            weights_map[fold]["TPE"] = tpe_path
            logging.info(f"Detected TPE best.pt for {fold} => {tpe_path}")
        else:
            logging.warning(f"Missing TPE best.pt for {fold} => {tpe_path}")

        # GP-BHO => "GP_BHO"
        gp_dir = f"GPSAMPLER_outer_{fold_idx}_inner_1"
        gp_path = os.path.join(root_weights_folder, gp_dir, "weights", "best.pt")
        if os.path.isfile(gp_path):
            weights_map[fold]["GP_BHO"] = gp_path
            logging.info(f"Detected GP-BHO best.pt for {fold} => {gp_path}")
        else:
            logging.warning(f"Missing GP-BHO best.pt for {fold} => {gp_path}")

        # Simulated Annealing
        sim_dir = f"SIMULATED_ANNEALING_outer_{fold_idx}_inner_1"
        sim_path = os.path.join(root_weights_folder, sim_dir, "weights", "best.pt")
        if os.path.isfile(sim_path):
            weights_map[fold]["Simulated_Annealing"] = sim_path
            logging.info(f"Detected Simulated Annealing best.pt for {fold} => {sim_path}")
        else:
            logging.warning(f"Missing Simulated Annealing best.pt for {fold} => {sim_path}")

        # Baseline
        base_dir = f"BASELINE_outer_{fold_idx}_inner_1"
        base_path = os.path.join(root_weights_folder, base_dir, "weights", "best.pt")
        if os.path.isfile(base_path):
            weights_map[fold]["Baseline"] = base_path
            logging.info(f"Detected Baseline best.pt for {fold} => {base_path}")
        else:
            logging.warning(f"Missing Baseline best.pt for {fold} => {base_path}")

    return fold_csv_paths, weights_map


def create_output_folders(output_root: str) -> Dict[str, str]:
    """
    Create 3 fold-named folders (Fold_1_inference, Fold_2_inference, Fold_3_inference),
    each containing subfolders [p100, p99, p90_98, p70_90, p50_70, nolesion].

    Args:
        output_root (str): The path where the fold subfolders will be created.

    Returns:
        Dict[str, str]: Mapping fold name to the folder path (e.g. fold_1 -> /.../Fold_1_inference).
    """
    fold_to_output_path = {}
    for i, fold in enumerate(FOLD_NAMES, start=1):
        fold_dir = os.path.join(output_root, f"Fold_{i}_inference")
        os.makedirs(fold_dir, exist_ok=True)
        for lesion_folder in LESION_FOLDERS:
            os.makedirs(os.path.join(fold_dir, lesion_folder), exist_ok=True)
        fold_to_output_path[fold] = fold_dir
        logging.info(f"Created {fold_dir} with subfolders {LESION_FOLDERS}")
    return fold_to_output_path


def run_inference_on_image(
    model_path: str,
    image_path: str,
    inference_params: Dict[str, Any]
) -> Any:
    """
    Runs YOLO inference on a single image.

    Args:
        model_path (str): Path to the YOLOv8 *.pt weights.
        image_path (str): Path to the input image.
        inference_params (Dict[str, Any]): YOLO inference parameters.

    Returns:
        Any: The inference result object from ultralytics model.predict().
    """
    model = YOLO(model_path)
    results = model.predict(source=image_path, **inference_params)
    return results


def draw_ground_truth_bboxes(
    image_path: str,
    groundtruth_path: str
) -> np.ndarray:
    """
    Draws ground-truth bounding boxes on the image (BGR) if groundtruth_path is not "nolesion".
    YOLO format: class x_center y_center width height (all normalized).
    - If class=0 => label="nolesion", else label=class.
    - Use color blue (255, 0, 0) in BGR.
    - Emulate YOLO's style: bounding box with a filled rectangle for the label.

    Args:
        image_path (str): Path to the image file.
        groundtruth_path (str): Path to the YOLO .txt file with bounding boxes, or "nolesion".

    Returns:
        np.ndarray: The BGR image with ground-truth bounding boxes drawn.
    """
    # Load the original image in BGR
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        logging.warning(f"Unable to load image for ground truth: {image_path}")
        return np.zeros((512, 512, 3), dtype=np.uint8)

    # If there's no groundtruth file (i.e., "nolesion"), then return the original
    if groundtruth_path.lower() == "nolesion":
        return image_bgr

    if not os.path.isfile(groundtruth_path):
        logging.warning(f"Groundtruth file not found: {groundtruth_path}")
        return image_bgr

    # Read the YOLO-format bounding boxes from the text file
    with open(groundtruth_path, 'r') as f:
        lines = f.readlines()

    img_h, img_w = image_bgr.shape[:2]

    for line in lines:
        # e.g. "0 0.583984 0.291992 0.117188 0.123047"
        values = line.strip().split()
        if len(values) != 5:
            continue

        class_id_str, x_center_str, y_center_str, w_str, h_str = values
        try:
            class_id = int(class_id_str)
            x_center = float(x_center_str)
            y_center = float(y_center_str)
            w = float(w_str)
            h = float(h_str)
        except ValueError:
            logging.warning(f"Skipping invalid groundtruth line: {line.strip()}")
            continue

        # Convert normalized coords to absolute pixel coords
        x_center_abs = x_center * img_w
        y_center_abs = y_center * img_h
        w_abs = w * img_w
        h_abs = h * img_h

        x1 = int(x_center_abs - w_abs / 2)
        y1 = int(y_center_abs - h_abs / 2)
        x2 = int(x_center_abs + w_abs / 2)
        y2 = int(y_center_abs + h_abs / 2)

        # If class=0 => label="nolesion", otherwise show numeric class
        if class_id == 0:
            label_text = "nolesion"
        else:
            label_text = str(class_id)

        # Draw the bounding box in blue
        box_color = (255, 0, 0)  # BGR => blue
        line_thickness = 2
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), box_color, thickness=line_thickness)

        # Draw a filled rectangle for the label
        # Estimate label background rectangle width
        label_bg_width = 10 + 9 * len(label_text)
        label_bg_height = 20
        # Above the top boundary of the box
        label_rect_top = y1 - label_bg_height if y1 - label_bg_height > 0 else y1
        label_rect_bottom = y1 if y1 - label_bg_height > 0 else y1 + label_bg_height

        cv2.rectangle(
            image_bgr,
            (x1, label_rect_top),
            (x1 + label_bg_width, label_rect_bottom),
            box_color,
            thickness=-1
        )

        # Put text in white
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        cv2.putText(
            image_bgr,
            label_text,
            (x1 + 5, label_rect_bottom - 5),
            font_face,
            font_scale,
            font_color,
            thickness=1,
            lineType=cv2.LINE_AA
        )

    return image_bgr


def create_1row_subplots(
    images_bgr: List[Any],
    titles: List[str]
) -> Any:
    """
    Creates a 1-row subplot with each image side-by-side.
    Converts BGR (OpenCV) images into RGB for display with Matplotlib.

    Args:
        images_bgr (List[Any]): A list of OpenCV images (BGR).
        titles (List[str]): Titles for each subplot.

    Returns:
        Any: A Matplotlib figure with the row of images.
    """
    num_models = len(images_bgr)
    fig, axes = plt.subplots(nrows=1, ncols=num_models, figsize=(6*num_models, 6))
    
    if num_models == 1:
        axes = [axes]

    for idx, (ax, img) in enumerate(zip(axes, images_bgr)):
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(titles[idx])
        ax.axis('off')

    fig.tight_layout()
    return fig


def pipeline_main(
    root_csv_folder: str,
    root_weights_folder: str,
    output_root: str
) -> None:
    """
    High-level pipeline that:
      1) Builds resource dictionaries for CSV and weight paths.
      2) Creates output folders.
      3) Iterates over each fold's test.csv.
      4) For each row (image) in test.csv:
         - Reads the 'LesionLabel' column to parse lesion folders.
         - Draws ground truth bounding boxes (first subplot column).
         - For each of the 4 models, runs inference and overlays bounding boxes.
         - Creates a 1×(1+4) subplot => [GroundTruth, TPE, GP-BHO, Simulated_Annealing, Baseline].
         - Saves that subplot in the correct lesion label folder(s).

    Args:
        root_csv_folder (str): Path to 'fold_1', 'fold_2', 'fold_3' directories, each with a test.csv.
        root_weights_folder (str): Path to the YOLO model subfolders with best.pt files.
        output_root (str): Path to create fold-wise and lesion-wise output.
    """
    # Step 1: Build dictionary resources
    fold_csv_map, weights_map = build_resources_dict(
        root_csv_folder=root_csv_folder,
        root_weights_folder=root_weights_folder
    )

    # Step 2: Create the 3 fold output folders
    fold_output_map = create_output_folders(output_root=output_root)

    # Step 3: Iterate over each fold's test.csv
    for fold_name, csv_path in fold_csv_map.items():
        logging.info(f"Processing {fold_name} => {csv_path}")
        if not os.path.isfile(csv_path):
            logging.warning(f"Missing test.csv for {fold_name}, skipping.")
            continue

        df = pd.read_csv(csv_path)  # columns: [LesionLabel, Frame_path, Groundtruth_path]

        # Step 4: For each row (image)
        for idx, row in df.iterrows():
            # 1. Parse the lesion labels from LesionLabel
            lesion_label_str = row["LesionLabel"]
            if "'" in lesion_label_str:
                lesion_labels = [
                    lbl.strip().replace("'", "")
                    for lbl in lesion_label_str.split(",")
                    if lbl.strip()
                ]
            else:
                lesion_labels = [lesion_label_str.strip()]

            # 2. Get the image path
            image_path = row["Frame_path"]
            if not os.path.isfile(image_path):
                logging.warning(f"No valid image path found for row {idx}: {image_path}. Skipping.")
                continue

            # 3. Get the groundtruth path
            groundtruth_path = row["Groundtruth_path"]  # e.g. /path/to/labels/xxx.txt or "nolesion"

            # 4. Draw ground truth bounding boxes as the first subplot image
            ground_truth_bgr = draw_ground_truth_bboxes(image_path, groundtruth_path)

            # Prepare for subplot creation: first column is GT, next columns are model predictions
            subplot_images = [ground_truth_bgr]
            subplot_titles = ["GroundTruth"]

            # Display name mapping
            MODEL_DISPLAY_NAMES = {
                "TPE": "Tree-structured Parzen Estimator",
                "GP_BHO": "Gaussian Process-based Optimizer",
                "Simulated_Annealing": "Simulated Annealing",
                "Baseline": "Baseline",
            }

            # For detection-checking
            total_detections = 0  

            # 5. For each model in [TPE, GP_BHO, Simulated_Annealing, Baseline]
            model_paths_for_fold = weights_map[fold_name]
            for model_key, model_bestpt in model_paths_for_fold.items():
                # Check if the model path is valid
                if not model_bestpt or not os.path.isfile(model_bestpt):
                    logging.warning(
                        f"Missing model file for {model_key} in {fold_name}. Skipping model."
                    )
                    continue

                # Run inference
                results = run_inference_on_image(
                    model_path=model_bestpt,
                    image_path=image_path,
                    inference_params=INFERENCE_PARAMS
                )

                if not results:
                    logging.warning(f"Inference returned no results for {model_key}")
                    continue

                # The first (and only) result
                result = results[0]
                detections_count = 0
                if result.boxes is not None:
                    detections_count = len(result.boxes)
                    total_detections += detections_count
                
                annotated_rgb = result.plot(**VISUALIZATION_PARAMS)
                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

                subplot_images.append(annotated_bgr)
                display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
                subplot_titles.append(display_name)

            # Now we have ground_truth_bgr + 4 possible model results
            # Condition:
            #   1) groundtruth_path != "nolesion"
            #   2) total_detections == 0 (i.e., all models had zero boxes)
            if groundtruth_path != "nolesion" and total_detections == 0:
                # => skip saving and continue to the next row
                logging.info(
                    f"Skipping saving for row {idx} because groundtruth exists but no predictions from any model."
                )
                plt.close('all')
                continue

            if len(subplot_images) < 2:
                # Means we didn't get any inference images
                logging.warning(f"No subplot images for row {idx} => skipping.")
                continue

            logging.info(f"Total detections for row {idx}: {total_detections}")

            # 6. Create subplot
            fig = create_1row_subplots(
                images_bgr=subplot_images,
                titles=subplot_titles
            )

            # 7. Build the output path for each label
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            for lesion_label in lesion_labels:
                # If not in known folders, default to "nolesion"
                if lesion_label not in LESION_FOLDERS:
                    lesion_label = "nolesion"

                output_fold_dir = fold_output_map[fold_name]  # e.g. /.../Fold_1_inference
                out_folder = os.path.join(output_fold_dir, lesion_label)
                out_filename = f"{base_name}_subplot.pdf"
                out_path = os.path.join(out_folder, out_filename)

                fig.savefig(out_path, dpi=300, format="pdf", bbox_inches='tight')
                logging.info(f"Saved subplot for row {idx} => {out_path}")

            plt.close(fig)  # free memory

    logging.info("Pipeline completed successfully.")


def main() -> None:
    """
    Main function to trigger the pipeline. 
    Update the following paths as necessary.
    """
    root_csv_folder = "/media/hddb/mario/data/double_cv_splits"
    root_weights_folder = "/home/mariopascual/Projects/CADICA/CROSS_VALIDATION/runs/detect"
    output_root = "/home/mariopascual/Projects/CADICA/CROSS_VALIDATION/inference/results"
    
    pipeline_main(
        root_csv_folder=root_csv_folder,
        root_weights_folder=root_weights_folder,
        output_root=output_root
    )


if __name__ == "__main__":
    main()
