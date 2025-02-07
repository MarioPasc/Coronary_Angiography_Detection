#!/usr/bin/env python3
# coding: utf-8
"""
Multi-Image YOLOv8 Inference Script – Saving Each Inference Image Separately

For each dictionary in the provided images_info list (with keys:
  { "fold_name", "patient_id", "video_id", "frame_id" }):
  1. Find the matching row in <fold_name>/test.csv using substring "p{patient_id}_{video_id}_{frame_id}".
  2. Draw the ground truth bounding boxes if available.
  3. Run inference using the four model weights (TPE, GP-BHO, Simulated_Annealing, Baseline).
  4. Save separate PDF files for:
       - Ground Truth (filename suffix "_GT")
       - Each inference prediction (filenames include the corresponding model key)
     The naming convention is: {fold}_{patient}_{video}_{frame}_{model}.pdf
     (No title is added to the images.)
"""

import os
import logging
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
import string
from typing_extensions import TypedDict

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("Please install ultralytics >= 8.0.0") from e

# ---------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------
logging.basicConfig(
    filename="multi_image_inference.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# ---------------------------------------------------------------------
# YOLO Inference Config
# ---------------------------------------------------------------------
INFERENCE_PARAMS: Dict[str, Any] = {
    "conf": 0.25,
    "iou": 0.5,
    "imgsz": 512,
    "half": False,
    "device": "cuda:0",
    "max_det": 300,
    "vid_stride": 1,
    "stream_buffer": False,
    "visualize": False,
    "augment": False,
    "agnostic_nms": False,
    "classes": None,
    "retina_masks": False,
    "embed": None,
    "project": None,
    "name": None,
    "show": False,
    "save": False,
    "save_frames": False,
    "save_txt": False,
    "save_conf": False,
    "save_crop": False,
    "show_labels": True,
    "show_conf": True,
    "show_boxes": True,
    "line_width": None,
}

VISUALIZATION_PARAMS: Dict[str, Any] = {
    "conf": True,
    "line_width": 1,
    "font_size": 12,
    "font": "Helvetica.ttf",
    "pil": False,
    "img": None,
    "im_gpu": None,
    "kpt_radius": 3,
    "kpt_line": True,
    "labels": True,
    "boxes": True,
    "probs": False,
    "show": False,
    "color_mode": "class",
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
    model_path: str, image_path: str, inference_params: Dict[str, Any]
) -> Any:
    """
    Runs YOLO inference on a single image and returns the result object.

    :param model_path: Path to the YOLO model weights (.pt).
    :param image_path: Path to the image on which inference is performed.
    :param inference_params: Dictionary of inference hyperparameters.
    :return: YOLO inference results (list of Result objects).
    """
    model = YOLO(model_path)
    results = model.predict(source=image_path, **inference_params)
    return results

def draw_ground_truth_bboxes(image_path: str, groundtruth_path: str) -> np.ndarray:
    """
    Draw YOLO-format bounding boxes from `groundtruth_path` onto an image (BGR).
    If groundtruth_path == 'nolesion' or does not exist, returns the original image.

    :param image_path: Path to the image on which to draw the GT boxes.
    :param groundtruth_path: Path to the YOLO .txt with GT boxes (or "nolesion").
    :return: BGR image (NumPy array) with drawn boxes, or unmodified if no GT.
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        logging.warning(f"Cannot load image: {image_path}")
        return np.zeros((512, 512, 3), dtype=np.uint8)

    if groundtruth_path.lower() == "nolesion":
        return image_bgr

    if not os.path.isfile(groundtruth_path):
        logging.warning(f"Groundtruth file not found: {groundtruth_path}")
        return image_bgr

    img_h, img_w = image_bgr.shape[:2]
    with open(groundtruth_path, "r") as f:
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

        x1 = int(x_c - w / 2)
        y1 = int(y_c - h / 2)
        x2 = int(x_c + w / 2)
        y2 = int(y_c + h / 2)

        label_text = "lesion" if cls_id == 0 else str(cls_id)
        box_color = (255, 0, 0)  # Blue in BGR
        thickness = 2
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), box_color, thickness=thickness)

        label_bg_width = 10 + 9 * len(label_text)
        label_bg_height = 20
        label_rect_top = max(y1 - label_bg_height, 0)
        label_rect_bottom = y1 if (y1 - label_bg_height) > 0 else (y1 + label_bg_height)

        cv2.rectangle(
            image_bgr,
            (x1, label_rect_top),
            (x1 + label_bg_width, label_rect_bottom),
            box_color,
            thickness=-1,
        )
        cv2.putText(
            image_bgr,
            label_text,
            (x1 + 5, label_rect_bottom - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return image_bgr

def build_resources_dict(
    root_csv_folder: str, root_weights_folder: str
) -> Tuple[Dict[str, str], FoldWeightsMap]:
    """
    Creates two structures:
      1) fold_csv_paths: { "fold_1": <test.csv path>, "fold_2": ... }
      2) weights_map (FoldWeightsMap): nested dict of fold→{ TPE, GP_BHO, Simulated_Annealing, Baseline }

    :param root_csv_folder: Base folder where fold_1/fold_2/fold_3 subfolders each contain test.csv.
    :param root_weights_folder: Base folder where model weights reside.
    :return: (fold_csv_paths, weights_map)
    """
    FOLD_NAMES = ["fold_1", "fold_2", "fold_3"]

    fold_csv_paths: Dict[str, str] = {}
    weights_map: FoldWeightsMap = {
        "fold_1": {"TPE": "", "GP_BHO": "", "Simulated_Annealing": "", "Baseline": ""},
        "fold_2": {"TPE": "", "GP_BHO": "", "Simulated_Annealing": "", "Baseline": ""},
        "fold_3": {"TPE": "", "GP_BHO": "", "Simulated_Annealing": "", "Baseline": ""},
    }

    # Find CSV files for each fold
    for fold in FOLD_NAMES:
        csv_path = os.path.join(root_csv_folder, fold, "test.csv")
        if os.path.isfile(csv_path):
            fold_csv_paths[fold] = csv_path
        else:
            logging.warning(f"No CSV found for {fold}: {csv_path}")

    # Find model weight files for each fold
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
# New Function: Save Each Inference Image Separately
# ---------------------------------------------------------------------
def multi_image_inference_separate(
    images_info: List[Dict[str, str]],
    fold_csv_map: Dict[str, str],
    weights_map: FoldWeightsMap,
    output_folder: str,
) -> None:
    """
    Performs inference on multiple images (specified by a list of dictionaries),
    each having { "fold_name", "patient_id", "video_id", "frame_id" }.
    For each image, this function saves separate PDF files for:
      - Ground Truth (with drawn GT boxes, if available) as a file with suffix "GT"
      - Inference predictions for each model (TPE, GP_BHO, Simulated_Annealing, Baseline)
    The output filename format is:
      {fold}_{patient}_{video}_{frame}_{model}.pdf
    No title is added to any saved image.
    
    :param images_info: List of dictionaries specifying each image to process.
    :param fold_csv_map: Dictionary mapping each fold to its CSV path.
    :param weights_map: Nested dictionary with model weight paths for each fold.
    :param output_folder: Directory where PDF files will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    MODEL_KEYS_IN_ORDER = ["TPE", "GP_BHO", "Simulated_Annealing", "Baseline"]

    for info in images_info:
        fold_name = info["fold_name"]
        patient_id = info["patient_id"]
        video_id = info["video_id"]
        frame_id = info["frame_id"]

        # Retrieve the CSV file for the given fold.
        csv_path = fold_csv_map.get(fold_name, "")
        if not csv_path or not os.path.isfile(csv_path):
            logging.warning(f"CSV for fold '{fold_name}' not found. Skipping image {patient_id}_{video_id}_{frame_id}.")
            continue

        # Read CSV and find the matching row.
        df = pd.read_csv(csv_path)
        search_str = f"{patient_id}_{video_id}_{frame_id}"
        subset = df[df["Frame_path"].str.contains(search_str, na=False)]
        if subset.empty:
            logging.warning(f"No entries found in {csv_path} for substring '{search_str}'. Skipping image.")
            continue

        row = subset.iloc[0]
        image_path = row["Frame_path"]
        groundtruth_path = row["Groundtruth_path"]

        if not os.path.isfile(image_path):
            logging.warning(f"Image file not found: {image_path}. Skipping image.")
            continue

        # -------------------------------
        # Save Ground Truth Image
        # -------------------------------
        gt_bgr = draw_ground_truth_bboxes(image_path, groundtruth_path)
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        gt_filename = f"{fold_name}_{patient_id}_{video_id}_{frame_id}_GT.pdf"
        gt_out_path = os.path.join(output_folder, gt_filename)
        fig, ax = plt.subplots()
        ax.imshow(gt_rgb)
        ax.axis("off")  # Remove axis and title
        fig.savefig(gt_out_path, dpi=300, format="pdf", bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Saved ground truth image to {gt_out_path}")

        # -------------------------------
        # Save Inference Prediction Images
        # -------------------------------
        fold_model_paths = weights_map.get(fold_name, {})
        for model_key in MODEL_KEYS_IN_ORDER:
            weight_path = fold_model_paths.get(model_key, "")
            if not weight_path or not os.path.isfile(weight_path):
                logging.warning(f"Weight not found for {model_key} in {fold_name}. Skipping inference for this model.")
                continue

            results = run_inference_on_image(weight_path, image_path, INFERENCE_PARAMS)
            if not results or len(results) == 0:
                logging.warning(f"No inference results for {model_key} on image {patient_id}_{video_id}_{frame_id}.")
                continue

            # Get annotated image with predictions
            annotated_rgb = results[0].plot(**VISUALIZATION_PARAMS)
            pred_filename = f"{fold_name}_{patient_id}_{video_id}_{frame_id}_{model_key}.pdf"
            pred_out_path = os.path.join(output_folder, pred_filename)
            fig, ax = plt.subplots()
            ax.imshow(annotated_rgb)
            ax.axis("off")
            fig.savefig(pred_out_path, dpi=300, format="pdf", bbox_inches="tight")
            plt.close(fig)
            logging.info(f"Saved prediction image for {model_key} to {pred_out_path}")

def main():
    """
    Example usage:
      1. Build the fold_csv_map and weights_map from provided paths.
      2. Provide a list of dictionaries for images_info.
      3. Save each resulting image (GT and predictions) as a separate PDF file.
    """
    # Adjust these as needed.
    root_csv_folder = "/media/hddb/mario/data/double_cv_splits"
    root_weights_folder = "/home/mariopascual/Projects/CADICA/CROSS_VALIDATION/runs/detect"
    output_folder = "/home/mariopascual/Projects/CADICA/CROSS_VALIDATION/inference/multi_image_inference"

    # 1. Build dictionary resources.
    fold_csv_map, weights_map = build_resources_dict(root_csv_folder, root_weights_folder)

    # 2. Specify the images to process.
    images_to_process = [
        {
            "fold_name": "fold_1",
            "patient_id": "p31",
            "video_id": "v12",
            "frame_id": "00029",
        },
        {
            "fold_name": "fold_3",
            "patient_id": "p12",
            "video_id": "v8",
            "frame_id": "00034",
        },
        {
            "fold_name": "fold_2",
            "patient_id": "p30",
            "video_id": "v5",
            "frame_id": "00029",
        },
        {
            "fold_name": "fold_2",
            "patient_id": "p30",
            "video_id": "v1",
            "frame_id": "00018",
        },
    ]

    # 3. Run inference and save each result as a separate PDF.
    multi_image_inference_separate(
        images_info=images_to_process,
        fold_csv_map=fold_csv_map,
        weights_map=weights_map,
        output_folder=output_folder,
    )

if __name__ == "__main__":
    main()
