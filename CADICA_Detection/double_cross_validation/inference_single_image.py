#!/usr/bin/env python3
# coding: utf-8
"""
Single-Image YOLOv8 Inference Script, using your multi-fold CSV/weight structure.

Given:
  - fold_name (e.g. "fold_1")
  - patient_id (e.g. "p13")
  - video_id   (e.g. "v5")
  - frame_id   (e.g. "00026")

We find the row in fold_name's test.csv whose 'Frame_path'
contains the substring "p13_v5_00026".
Then we run inference on that one image using all 4 model weights
(TPE, GP-BHO, Simulated_Annealing, Baseline), and produce a 1-row subplot:
  [GroundTruth | TPE | GP-BHO | Simulated_Annealing | Baseline]

The result is saved to an output folder as a PDF.
"""

import os
import logging
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from typing_extensions import TypedDict

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("Please install ultralytics >= 8.0.0") from e

# ---------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------
logging.basicConfig(
    filename='single_image_inference.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# ---------------------------------------------------------------------
# YOLO Inference Config
# ---------------------------------------------------------------------
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
    'save': False,
    'save_frames': False,
    'save_txt': False,
    'save_conf': False,
    'save_crop': False,
    'show_labels': True,
    'show_conf': True,
    'show_boxes': True,
    'line_width': None,
}

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

# ---------------------------------------------------------------------
# TypedDict for fold→model weights
# ---------------------------------------------------------------------
class WeightsPathMap(TypedDict):
    TPE: str
    GP_BHO: str
    Simulated_Annealing: str
    Baseline: str

class FoldWeightsMap(TypedDict):
    fold_1: WeightsPathMap
    fold_2: WeightsPathMap
    fold_3: WeightsPathMap


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def run_inference_on_image(
    model_path: str,
    image_path: str,
    inference_params: Dict[str, Any]
) -> Any:
    """Runs YOLO inference on a single image and returns the result."""
    model = YOLO(model_path)
    results = model.predict(source=image_path, **inference_params)
    return results

def draw_ground_truth_bboxes(
    image_path: str,
    groundtruth_path: str
) -> np.ndarray:
    """
    Draw YOLO-format bounding boxes onto an image (BGR).
    If groundtruth_path == 'nolesion' or does not exist, return the original image.
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        logging.warning(f"Cannot load image: {image_path}")
        return np.zeros((512, 512, 3), dtype=np.uint8)

    # If "nolesion", there's no bounding box to draw
    if groundtruth_path.lower() == "nolesion":
        return image_bgr

    if not os.path.isfile(groundtruth_path):
        logging.warning(f"Groundtruth file not found: {groundtruth_path}")
        return image_bgr

    img_h, img_w = image_bgr.shape[:2]

    with open(groundtruth_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        vals = line.strip().split()
        if len(vals) != 5:
            continue
        cls_id_str, x_str, y_str, w_str, h_str = vals

        try:
            cls_id = int(cls_id_str)
            x_c = float(x_str) * img_w
            y_c = float(y_str) * img_h
            w = float(w_str) * img_w
            h = float(h_str) * img_h
        except ValueError:
            logging.warning(f"Skipping invalid GT line: {line.strip()}")
            continue

        x1 = int(x_c - w/2)
        y1 = int(y_c - h/2)
        x2 = int(x_c + w/2)
        y2 = int(y_c + h/2)

        # If class=0 => label "lesion", else numeric
        label_text = "lesion" if cls_id == 0 else str(cls_id)

        box_color = (255, 0, 0)  # BGR: blue
        thickness = 2
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), box_color, thickness=thickness)

        # Label background
        label_bg_width = 10 + 9*len(label_text)
        label_bg_height = 20
        label_rect_top = max(y1 - label_bg_height, 0)
        label_rect_bottom = y1 if (y1 - label_bg_height) > 0 else (y1 + label_bg_height)

        cv2.rectangle(
            image_bgr,
            (x1, label_rect_top),
            (x1 + label_bg_width, label_rect_bottom),
            box_color, thickness=-1
        )

        cv2.putText(
            image_bgr, label_text,
            (x1 + 5, label_rect_bottom - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
            thickness=1, lineType=cv2.LINE_AA
        )

    return image_bgr

def create_1row_subplots(
    images_bgr: List[np.ndarray],
    titles: List[str]
) -> Any:
    """Given a list of BGR images and titles, produce a 1-row Matplotlib figure (RGB display)."""
    n = len(images_bgr)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))

    if n == 1:
        axes = [axes]

    for ax, img_bgr, title in zip(axes, images_bgr, titles):
        # Convert to RGB for matplotlib
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None
        ax.imshow(img_rgb)
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Resources-building (Optional, If You Already Have a Folding Setup)
# ---------------------------------------------------------------------
def build_resources_dict(
    root_csv_folder: str,
    root_weights_folder: str
) -> Tuple[Dict[str, str], FoldWeightsMap]:
    """
    Creates a dictionary: { "fold_1": <csv_path>, ... }
    and a dictionary mapping each fold to the 4 best.pt model paths: TPE, GP_BHO, etc.
    """
    FOLD_NAMES = ["fold_1", "fold_2", "fold_3"]

    fold_csv_paths: Dict[str, str] = {}
    weights_map: FoldWeightsMap = {
        "fold_1": {"TPE": "", "GP_BHO": "", "Simulated_Annealing": "", "Baseline": ""},
        "fold_2": {"TPE": "", "GP_BHO": "", "Simulated_Annealing": "", "Baseline": ""},
        "fold_3": {"TPE": "", "GP_BHO": "", "Simulated_Annealing": "", "Baseline": ""},
    }

    # Find CSVs
    for fold in FOLD_NAMES:
        csv_path = os.path.join(root_csv_folder, fold, "test.csv")
        if os.path.isfile(csv_path):
            fold_csv_paths[fold] = csv_path
        else:
            logging.warning(f"No CSV found for {fold}: {csv_path}")

    # Find best.pt for each fold & model
    for idx, fold in enumerate(FOLD_NAMES, start=1):
        # TPE
        tpe_dir = f"TPE_outer_{idx}_inner_1"
        tpe_path = os.path.join(root_weights_folder, tpe_dir, "weights", "best.pt")
        if os.path.isfile(tpe_path):
            weights_map[fold]["TPE"] = tpe_path

        # GP-BHO
        gp_dir = f"GPSAMPLER_outer_{idx}_inner_1"
        gp_path = os.path.join(root_weights_folder, gp_dir, "weights", "best.pt")
        if os.path.isfile(gp_path):
            weights_map[fold]["GP_BHO"] = gp_path

        # Simulated Annealing
        sim_dir = f"SIMULATED_ANNEALING_outer_{idx}_inner_1"
        sim_path = os.path.join(root_weights_folder, sim_dir, "weights", "best.pt")
        if os.path.isfile(sim_path):
            weights_map[fold]["Simulated_Annealing"] = sim_path

        # Baseline
        base_dir = f"BASELINE_outer_{idx}_inner_1"
        base_path = os.path.join(root_weights_folder, base_dir, "weights", "best.pt")
        if os.path.isfile(base_path):
            weights_map[fold]["Baseline"] = base_path

    return fold_csv_paths, weights_map


# ---------------------------------------------------------------------
# Main Single-Image Inference Function
# ---------------------------------------------------------------------
def run_single_image_inference(
    fold_name: str,
    patient_id: str,
    video_id: str,
    frame_id: str,
    fold_csv_map: Dict[str, str],
    weights_map: FoldWeightsMap,
    output_folder: str
):
    """
    1. Load fold_name's test.csv.
    2. Search for row(s) whose Frame_path contains 'p{patient_id}_v{video_id}_{frame_id}'.
    3. For that row, get Frame_path + Groundtruth_path.
    4. Draw GT boxes, run YOLO inferences for 4 models, create 1-row subplot.
    5. Save as PDF in output_folder.

    Example usage:
        run_single_image_inference(
           fold_name="fold_1", patient_id="p13", video_id="v5", frame_id="00026",
           fold_csv_map=some_csv_dict, weights_map=some_weights_dict,
           output_folder="..."
        )
    """
    # 1. Retrieve test.csv path for this fold
    csv_path = fold_csv_map.get(fold_name, "")
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"CSV file for {fold_name} not found.")
        return

    # 2. Read the CSV, filter by 'p{patient}_v{video}_{frame}'
    df = pd.read_csv(csv_path)

    search_str = f"{patient_id}_{video_id}_{frame_id}"
    subset = df[df["Frame_path"].str.contains(search_str, na=False)]
    if len(subset) == 0:
        logging.warning(f"No entries found in {csv_path} for substring '{search_str}'.")
        return

    # We assume exactly one row matches. If multiple match, we just take the first.
    row = subset.iloc[0]
    image_path = row["Frame_path"]
    groundtruth_path = row["Groundtruth_path"]  # e.g. nolesion or .txt path

    if not os.path.isfile(image_path):
        logging.error(f"Image file not found: {image_path}")
        return

    # 3. Draw ground-truth bounding boxes
    gt_image_bgr = draw_ground_truth_bboxes(image_path, groundtruth_path)

    # 4. For each of the 4 models, run inference
    MODEL_DISPLAY_NAMES = {
        "TPE": "Tree-structured Parzen Estimator",
        "GP_BHO": "Gaussian Process-based Optimizer",
        "Simulated_Annealing": "Simulated Annealing",
        "Baseline": "Baseline",
    }

    subplot_images = [gt_image_bgr]
    subplot_titles = ["GroundTruth"]

    model_paths = weights_map[fold_name]
    for model_key, weight_path in model_paths.items():
        if not weight_path or not os.path.isfile(weight_path):
            # Not all folds have all model weights, or you might have empty placeholders
            logging.warning(f"Skipping {model_key} in {fold_name} - weight file missing.")
            continue

        results = run_inference_on_image(weight_path, image_path, INFERENCE_PARAMS)
        if not results or len(results) == 0:
            logging.warning(f"No results from inference on {model_key} model.")
            continue

        # Each results list has exactly one item for a single image
        annotated_rgb = results[0].plot(**VISUALIZATION_PARAMS)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        subplot_images.append(annotated_bgr)
        subplot_titles.append(MODEL_DISPLAY_NAMES.get(model_key, model_key))

    if len(subplot_images) == 1:
        # Means we got only the GT image, no model results
        logging.warning("No model predictions to display. Possibly all weights missing?")
        return

    # 5. Create subplot and save
    fig = create_1row_subplots(subplot_images, subplot_titles)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_filename = f"{base_name}_subplot.pdf"  # e.g. p13_v5_00026_subplot.pdf
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, out_filename)

    fig.savefig(out_path, dpi=300, format='pdf', bbox_inches='tight')
    plt.close(fig)

    logging.info(f"Saved single-image inference subplot to {out_path}")


# ---------------------------------------------------------------------
# Demonstration main()
# ---------------------------------------------------------------------
def main():
    """
    Example main() usage:
      1. Build the fold_csv_map and weights_map.
      2. Choose a fold name, plus patient/video/frame to filter from that fold's CSV.
      3. Save the resulting subplot to some output folder.
    """
    # Adjust these as needed:
    root_csv_folder = "/media/hddb/mario/data/double_cv_splits"
    root_weights_folder = "/home/mariopascual/Projects/CADICA/CROSS_VALIDATION/runs/detect"
    output_folder = "/home/mariopascual/Projects/CADICA/CROSS_VALIDATION/inference/single_image_inference"

    # We'll demonstrate picking fold_1, plus p13, v5, 00026
    fold_name = "fold_1"
    patient_id = "p13"
    video_id = "v5"
    frame_id = "00026"  # from your example p13_v5_00026

    # 1. Build dictionary resources
    fold_csv_map, weights_map = build_resources_dict(root_csv_folder, root_weights_folder)

    # 2. Perform single image inference
    run_single_image_inference(
        fold_name=fold_name,
        patient_id=patient_id,
        video_id=video_id,
        frame_id=frame_id,
        fold_csv_map=fold_csv_map,
        weights_map=weights_map,
        output_folder=output_folder
    )


if __name__ == "__main__":
    main()
