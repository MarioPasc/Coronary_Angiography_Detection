import os
import yaml
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from ultralytics import YOLO
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

import json
import pandas as pd
import os
from PIL import Image

def convert_to_coco_format(csv_path, gt_json_path, dt_json_path):
    df = pd.read_csv(csv_path)
    
    images = []
    annotations = []
    detections = []
    categories = []
    
    category_name_to_id = {}
    annotation_id = 1  # Start annotation IDs from 1
    detection_id = 1   # Start detection IDs from 1
    image_id = 1       # Start image IDs from 1

    for idx, row in df.iterrows():
        image_path = row['Image']
        image_name = os.path.basename(image_path)
        
        # Open image to get dimensions
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Add image info
        image_info = {
            'id': image_id,
            'file_name': image_name,
            'width': width,
            'height': height
        }
        images.append(image_info)
        
        # Process ground truth annotations
        gt_labels = eval(row['GT_labels'])  # Convert string representation of list to actual list
        gt_bboxes = eval(row['GT_bboxs'])   # Convert string representation of list to actual list
        
        for gt_label, gt_bbox in zip(gt_labels, gt_bboxes):
            # Get or assign category ID
            if gt_label not in category_name_to_id:
                category_id = len(category_name_to_id) + 1  # IDs start from 1
                category_name_to_id[gt_label] = category_id
                categories.append({'id': category_id, 'name': gt_label})
            else:
                category_id = category_name_to_id[gt_label]
            
            # Convert normalized bbox to absolute pixel values and COCO format
            x_center, y_center, bbox_width, bbox_height = map(float, gt_bbox.strip().split())
            x_min = (x_center - bbox_width / 2) * width
            y_min = (y_center - bbox_height / 2) * height
            bbox_width = bbox_width * width
            bbox_height = bbox_height * height
            
            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [x_min, y_min, bbox_width, bbox_height],
                'area': bbox_width * bbox_height,
                'iscrowd': 0
            }
            annotations.append(annotation)
            annotation_id += 1
        
        # Process predicted detections
        pred_bboxes = eval(row['Pred_bboxs'])  # This should be a list of lists
        pred_confs = eval(row['Pred_confs'])   # List of confidence scores

        for pred_bbox, score in zip(pred_bboxes, pred_confs):
            # Assuming that the category for predictions is the same as ground truth
            # If your model predicts category IDs, use that instead
            # Here we will assign a default category ID, as we might not have category info in predictions
            # Modify this part based on your actual predicted categories
            category_id = 1  # If you have multiple categories, adjust accordingly
            
            # Convert bbox to COCO format
            # Your pred_bbox is already in absolute pixel values, but may need to adjust
            # If pred_bbox is [x_center, y_center, width, height], convert it
            x_center, y_center, bbox_width, bbox_height = pred_bbox
            x_min = x_center - bbox_width / 2
            y_min = y_center - bbox_height / 2
            
            detection = {
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [x_min, y_min, bbox_width, bbox_height],
                'score': score
            }
            detections.append(detection)
            detection_id += 1
        
        image_id += 1

    # Save ground truth annotations to JSON
    gt_data = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
    with open(gt_json_path, 'w') as f:
        json.dump(gt_data, f, indent=4)
    
    # Save detections to JSON
    with open(dt_json_path, 'w') as f:
        json.dump(detections, f, indent=4)

    print(f"Ground truth annotations saved to {gt_json_path}")
    print(f"Detection results saved to {dt_json_path}")

# Example usage:
save_path = '/home/mariopasc/Python/Results/Coronariografias/Results_Paper/test'

csv_path = os.path.join(save_path, 'test_predictions.csv')
gt_json_path = os.path.join(save_path, 'ground_truth_annotations.json')
dt_json_path = os.path.join(save_path, 'detection_results.json')

config_path = './CADICA_Detection/model/config_labels.yaml'
model_path = '/home/mariopasc/Python/Results/Coronariografias/Results_Paper/iteration2/validation/ateroesclerosis_training/weights/best.pt'

convert_to_coco_format(csv_path, gt_json_path, dt_json_path)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load ground truth
coco_gt = COCO('ground_truth_annotations.json')

# Load detections
coco_dt = coco_gt.loadRes('detection_results.json')

# Initialize COCOeval object
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

# Evaluate
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()


