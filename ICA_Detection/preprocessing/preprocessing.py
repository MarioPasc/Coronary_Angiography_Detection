# ica_yolo_detection/preprocessing/preprocessing.py

import os
import cv2
import json
import shutil
from typing import List, Dict, Any
from tqdm import tqdm  # type: ignore
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# Example imports of your existing utility functions:
from ICA_Detection.tools.format_standarization import apply_format_standarization
from ICA_Detection.tools.dtype_standarization import apply_dtype_standarization
from ICA_Detection.tools.resolution import apply_resolution
from ICA_Detection.tools.fse import filtering_smoothing_equalization
from ICA_Detection.tools.clahe import clahe_enhancement
from ICA_Detection.tools.bbox_translation import common_to_yolo, rescale_bbox
from ICA_Detection.tools.dataset_conversions import (
    construct_yolo,
    construct_coco_detection,
    construct_coco_segmentation
)

DEBUG: bool = False

def process_images_by_task(
    json_path: str,
    out_dir: str,
    steps_order: List[str],
) -> None:
    """
    Process images and annotations based on the task type, 
    which is inferred from the top-level JSON key 
    ('Stenosis_Detection' or 'Arteries_Segmentation').

    Args:
        json_path (str): Path to the planned JSON file 
                         (with top-level keys 'Stenosis_Detection' 
                          or 'Arteries_Segmentation').
        out_dir (str): Directory where the processed data should be stored.
        steps_order (List[str]): List of preprocessing steps in the order 
                                 they should be applied.
    """
    # ---------------------------------------------------------------------
    # 1. Read JSON and identify the task by checking the top-level key
    # ---------------------------------------------------------------------
    with open(json_path, "r") as f:
        data = json.load(f)
    
    print(f"Processing JSON file: {Path(json_path)}")
    if len(data.keys()) == 0:
        print(f"The JSON file {Path(json_path).name} is empty. This is because we have no dataset for this task.")
        print("Hint: You probably choose CADICA or KEMEROVO, which do not have a segmentation task.")
        return
    
    # Figure out if this is detection or segmentation by examining the JSON root
    if "Stenosis_Detection" in data:
        dataset_dict = data["Stenosis_Detection"]
        task = "detection"
    elif "Arteries_Segmentation" in data:
        dataset_dict = data["Arteries_Segmentation"]
        task = "segmentation"
    else:
        raise ValueError(
            "No recognized top-level key found in JSON. "
            "Expected 'Stenosis_Detection' or 'Arteries_Segmentation'."
        )

    # ---------------------------------------------------------------------
    # 2. Create output folder structure depending on your use case
    # ---------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    images_out = os.path.join(out_dir, "images")
    labels_out = os.path.join(out_dir, "labels")

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    
    # ---------------------------------------------------------------------
    # 3. Iterate through each entry in the dataset and apply the pipeline
    # ---------------------------------------------------------------------
    for uid, entry in tqdm(dataset_dict.items(), desc="Processing JSON entries", colour="green"):

        # -----------------------------------------------------------------
        # 3a. Basic checks and paths
        # -----------------------------------------------------------------
        img_info = entry.get("image", {})
        orig_img_path = img_info.get("route")
        if not orig_img_path or not os.path.exists(orig_img_path):
            print(f"Skipping {uid}: original image path not found.")
            continue

        # We'll store the final processed image as PNG
        working_filename = f"{uid}.png"
        working_img_path = os.path.join(images_out, working_filename)

        # -----------------------------------------------------------------
        # 3b. Format Standardization
        # -----------------------------------------------------------------
        if (
            "format_standarization" in entry.get("preprocessing_plan", {})
            and "format_standarization" in steps_order
        ):
            desired_format = (
                entry["preprocessing_plan"]["format_standarization"]
                .get("desired_format", "png")
                .lower()
            )
            ret = apply_format_standarization(orig_img_path, working_img_path, desired_format)
            if ret is None:
                print(f"Error converting format for {uid}.")
                continue
        else:
            # Copy instead, if we are not converting
            shutil.copy2(orig_img_path, working_img_path)

        # Rename 'route' to 'original_route' and introduce 'dataset_route'
        img_info["original_route"] = img_info.pop("route")
        img_info["dataset_route"] = working_img_path

        current_img_path = working_img_path

        # -----------------------------------------------------------------
        # 3c. Dtype Standardization
        # -----------------------------------------------------------------
        if (
            "dtype_standarization" in entry.get("preprocessing_plan", {})
            and "dtype_standarization" in steps_order
        ):
            desired_dtype = entry["preprocessing_plan"]["dtype_standarization"].get("desired")
            new_img_path = current_img_path
            ret = apply_dtype_standarization(current_img_path, new_img_path, desired_dtype)
            if ret is None:
                print(f"Error converting dtype for {uid}.")
                continue

        # -----------------------------------------------------------------
        # 3d. Resolution Standardization + Bbox/Mask Rescaling
        # -----------------------------------------------------------------
        if (
            "resolution_standarization" in entry.get("preprocessing_plan", {})
            and "resolution_standarization" in steps_order
        ):
            res_plan = entry["preprocessing_plan"]["resolution_standarization"]
            desired_X = res_plan.get("desired_X", 512)
            desired_Y = res_plan.get("desired_Y", 512)
            method = res_plan.get("method", "bilinear")

            old_width = img_info.get("width")
            old_height = img_info.get("height")

            new_img_path = current_img_path  # Overwrite in place
            ret = apply_resolution(current_img_path, new_img_path, desired_X, desired_Y, method)
            if ret is None:
                print(f"Error resizing image for {uid}.")
                continue

            # Update the JSON's stored width/height
            img_info["width"] = desired_X
            img_info["height"] = desired_Y

            # Depending on the task, we may have different annotation fields
            annotations = entry.get("annotations", {})

            # ---------------- Segmentation branch ----------------
            if task == "segmentation":
                # Example: Arteries_Segmentation has "vessel_segmentations"
                if "vessel_segmentations" in annotations and old_width and old_height:
                    vessel_segmentations = annotations["vessel_segmentations"]

                    mask_filename = f"{uid}_seg.png"
                    mask_path = os.path.join(labels_out, mask_filename)

                    # Create an empty mask in the original size
                    mask = Image.new("L", (old_width, old_height), 0)
                    draw = ImageDraw.Draw(mask)

                    for vessel_seg in vessel_segmentations:
                        # If there's a bounding box, rescale it
                        if "bbox" in vessel_seg:
                            vessel_seg["bbox"] = rescale_bbox(
                                vessel_seg["bbox"],
                                old_width,
                                old_height,
                                desired_X,
                                desired_Y,
                            )

                        # If there's segmentation coordinates, draw them
                        # (Here we assume an array "segment0", "segment1", or something similar.)
                        # You can unify them if they are all in "xyxy".
                        for key_seg, seg_coords in vessel_seg.items():
                            if key_seg.startswith("segment"):
                                points_array = np.array(seg_coords).reshape(-1, 2)
                                points_tuples = [(x, y) for x, y in points_array]
                                draw.polygon(points_tuples, fill=255)

                    # Resize the mask to the final size
                    mask = mask.resize((desired_X, desired_Y), Image.Resampling.NEAREST)
                    mask.save(mask_path)

            # ---------------- Detection branch ----------------
            elif task == "detection":
                # Example: Stenosis_Detection has bounding boxes
                if "bbox1" in annotations and old_width and old_height:
                    # Possibly rescale a single bounding box or multiple
                    # This is an example if your detection JSON might have multiple bboxes
                    bbox = annotations["bbox1"]
                    annotations["bbox1"] = rescale_bbox(bbox, old_width, old_height, desired_X, desired_Y)

        # -----------------------------------------------------------------
        # 3e. CLAHE (if any)
        # -----------------------------------------------------------------
        if (
            "clahe" in entry.get("preprocessing_plan", {})
            and "clahe" in steps_order
        ):
            clahe_plan = entry["preprocessing_plan"]["clahe"]
            window_size = clahe_plan.get("window_size", 5)
            sigma = clahe_plan.get("sigma", 1.0)
            clip_limit = clahe_plan.get("clipLimit", 2.0)
            tile_grid_size = tuple(clahe_plan.get("tileGridSize", (8, 8)))

            img = cv2.imread(current_img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error reading image for CLAHE for {uid}.")
                continue

            enhanced = clahe_enhancement(img, window_size, sigma, clip_limit, tile_grid_size)
            cv2.imwrite(current_img_path, enhanced)

        # -----------------------------------------------------------------
        # 3f. Filtering Smoothing Equalization
        # -----------------------------------------------------------------
        if (
            "filtering_smoothing_equalization" in entry.get("preprocessing_plan", {})
            and "filtering_smoothing_equalization" in steps_order
        ):
            fse_plan = entry["preprocessing_plan"]["filtering_smoothing_equalization"]
            window_size = fse_plan.get("window_size", 5)
            sigma = fse_plan.get("sigma", 1.0)

            img = cv2.imread(current_img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error reading image for FSE for {uid}.")
                continue

            enhanced = filtering_smoothing_equalization(img, window_size, sigma)
            cv2.imwrite(current_img_path, enhanced)

        # -----------------------------------------------------------------
        # 3g. Save or convert bounding box annotations (detection task)
        # -----------------------------------------------------------------
        # TODO: Guardar también la segmentación en formato YOLO
        
        # We are going to modify this only for now to save a detection dataset instead of segmentation
        # in the segmentation task. This is because we are going to do some trials to apply
        # detection with mask-guided attention with the YOLO model, and we need to have the detection 
        # dataset
        # if task == "detection"
        if task == "detection" or task == "segmentation": 
            labels_formats = entry.get("preprocessing_plan", {}).get("dataset_formats", {})
            # If 'lesion' might be True or False in the detection context
            # we only save labels for entries that have at least one bbox
            # (which also implies 'lesion' is True).
            if DEBUG: print(f"DEBUG: UID: {uid}, Lesion: {entry.get('lesion')}") # DEBUG
            if entry.get("lesion", False):
                annotations = entry.get("annotations", {})
                if DEBUG: print(f"DEBUG: UID: {uid}, Annotations: {annotations}") # DEBUG
                label_filename = annotations.get("name", f"{uid}.txt")

                # Collect all bounding boxes named "bbox1", "bbox2", etc.
                # This will handle multiple bounding boxes in your JSON.
                pascal_lines = []
                yolo_lines = []

                # Image dimensions for YOLO normalization
                orig_width = img_info.get("width", 512)
                orig_height = img_info.get("height", 512)

                bbox_container = annotations
                if "stenosis" in annotations and isinstance(annotations["stenosis"], dict):
                    bbox_container = annotations["stenosis"]
                if DEBUG: print(f"DEBUG: UID: {uid}, Bbox Container: {bbox_container}") # New DEBUG line


                for ann_key, bbox in bbox_container.items():
                    if ann_key.startswith("bbox"):
                        if DEBUG: print(f"DEBUG: UID: {uid}, ann_key: {ann_key}, bbox: {bbox}") # DEBUG
                        # 1) Pascal VOC–style line: 
                        #    class_idx xmin ymin xmax ymax
                        xmin = bbox["xmin"]
                        ymin = bbox["ymin"]
                        xmax = bbox["xmax"]
                        ymax = bbox["ymax"]
                        pascal_line = f"0 {xmin} {ymin} {xmax} {ymax}"  # "0" as class ID
                        pascal_lines.append(pascal_line)

                        # 2) YOLO–style line:
                        #    class_idx x_center y_center width height (all normalized to [0,1] if you want standard YOLO)
                        if labels_formats.get("YOLO", False):
                            yolo_bbox = common_to_yolo(bbox, orig_width, orig_height)
                            yolo_line = (
                                f"0 {yolo_bbox['x_center']} {yolo_bbox['y_center']} "
                                f"{yolo_bbox['width']} {yolo_bbox['height']}"
                            )
                            yolo_lines.append(yolo_line)
                
                if DEBUG: print(f"DEBUG: UID: {uid}, Pascal Lines: {pascal_lines}") # DEBUG
                if DEBUG: print(f"DEBUG: UID: {uid}, YOLO Lines: {yolo_lines}") # DEBUG

                # Write Pascal VOC–style labels
                labels_pascal_out = os.path.join(labels_out, "pascal_voc")
                os.makedirs(labels_pascal_out, exist_ok=True)
                label_out_path = os.path.join(labels_pascal_out, label_filename)
                with open(label_out_path, "w") as f:
                    f.write("\n".join(pascal_lines))

                # Write YOLO–style labels if requested
                if labels_formats.get("YOLO", False):
                    # Create subfolder (if desired) to store YOLO labels
                    labels_yolo_out = os.path.join(labels_out, "yolo")
                    os.makedirs(labels_yolo_out, exist_ok=True)

                    label_yolo_out_path = os.path.join(labels_yolo_out, label_filename)
                    print(f"Saving YOLO labels to {label_yolo_out_path}")
                    with open(label_yolo_out_path, "w") as f_yolo:
                        f_yolo.write("\n".join(yolo_lines))


    # ---------------------------------------------------------------------
    # 4. Save an updated JSON with processed info
    # ---------------------------------------------------------------------
    processed_json_path = os.path.join(out_dir, "json", "processed.json")

    # Overwrite the relevant portion in the original data
    # so that we keep the same structure, just with updated fields.
    if task == "detection":
        data["Stenosis_Detection"] = dataset_dict
    else:
        data["Arteries_Segmentation"] = dataset_dict

    with open(processed_json_path, "w") as f:
        json.dump(data, f, indent=2)

    # ---------------------------------------------------------------------
    # 5. Optionally generate final datasets (COCO, YOLO, etc.)
    # ---------------------------------------------------------------------
    # Extract the config from the last entry or define it globally
    # (You could also unify how you read your 'preprocessing_plan'.)
    # This is just an example snippet:
    if len(dataset_dict) > 0:
        last_entry_key = list(dataset_dict.keys())[-1]
        config = dataset_dict[last_entry_key].get("preprocessing_plan", {})
        generate_datasets(root_folder=out_dir, config=config, json_path=processed_json_path, task = task)

    print(f"All images processed. Updated JSON saved to: {processed_json_path}")


def generate_datasets(root_folder: str, config: Dict[str, Any], json_path: str, task: str) -> None:
    """
    Look at config['dataset_formats'] to determine which dataset
    formats should be generated. Then call the corresponding function
    from dataset_conversions.
    """
    dataset_formats = config.get("dataset_formats", {})
    root_path = Path(root_folder).resolve()
    datasets_path = root_path / "datasets"
    datasets_path.mkdir(exist_ok=True)

    print(f"Building datasets: ")
    print(dataset_formats)

    if dataset_formats.get("YOLO", False):
        construct_yolo(root_path)
        # Cleanup (if you want to remove intermediate YOLO labels)
        yolo_labels_dir = os.path.join(root_path, "yolo")
        if os.path.isdir(yolo_labels_dir):
            shutil.rmtree(yolo_labels_dir)

    if (
        dataset_formats.get("COCO", False)
    ):
        if task == "segmentation":
            construct_coco_segmentation(json_path=json_path, root_folder=str(root_path))
        else:
            construct_coco_detection(json_path=json_path, root_folder=str(root_path))