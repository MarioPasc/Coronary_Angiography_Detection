import math
import sys
import time
import os
import json
import numpy as np

from typing import Dict, Any, List, Tuple

import torch
import torchvision.models.detection.mask_rcnn

from ICA_Detection.tasks.detection.utils import utils
from ICA_Detection.tasks.detection.utils.coco_eval import CocoEvaluator
from ICA_Detection.tasks.detection.utils.coco_utils import get_coco_api_from_dataset

DEBUG: bool = False


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    print_freq,
    scaler=None,
):
    """
    Trains one epoch with a RetinaNet model that returns (classification_loss, regression_loss).
    """
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        # Simple warmup
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for batch in metric_logger.log_every(data_loader, print_freq, header):

        images = []
        targets = []
        for sample in batch:
            images.append(sample[0])  # PIL/np array
            targets.append(sample[1])  # Nx5

        images = images.to(device)

        # Convert each target dict -> Nx5 for FocalLoss
        formatted_targets = []
        for t in targets:
            boxes = t["boxes"].to(device)
            labels = t["labels"].to(device)
            if boxes.shape[0] == 0:
                annotation_tensor = (
                    torch.zeros((1, 5), dtype=torch.float32, device=device) - 1
                )
            else:
                annotation_tensor = torch.cat([boxes, labels.unsqueeze(1)], dim=1)

            if annotation_tensor.shape[0] == 0:
                annotation_tensor = (
                    torch.zeros((1, 5), dtype=torch.float32, device=device) - 1
                )
            formatted_targets.append(annotation_tensor)

        with torch.amp.autocast(enabled=(scaler is not None), device_type="cuda"):
            classification_loss, regression_loss = model(images, formatted_targets)
            total_loss = classification_loss + regression_loss

        # Reduce for DDP if needed
        loss_dict = {
            "classification": classification_loss,
            "regression": regression_loss,
        }
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print(f"[ERROR] Loss is {loss_value}, stopping training.")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(
    model,
    data_loader,
    device,
    epoch,
    json_path=None,
    score_thresh=0.05,
    iou_thresh=0.5,
):
    """
    Runs inference on each batch using the model in eval mode.
    Your RetinaNet in eval mode returns:
        scores, class_indices, transformed_anchors
    for each image in the batch.

    We then feed these results into a COCO evaluator to measure mAP, etc.
    Optionally, we save a JSON with predictions & GT.

    If your model returns only a single set of (scores, classes, boxes)
    for the entire batch, we need to split them out per-image.
    Otherwise, if your model is coded to return a list of (scores, classes, boxes)
    for each image, we can just parse it directly.
    """

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Evaluation (epoch {epoch}):"

    # Prepare COCO evaluator
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    all_results = {}
    all_results[f"epoch_{epoch}"] = {}

    dataset_root = getattr(data_loader.dataset, "root", "")
    if not dataset_root:
        dataset_root = ""

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        # Move images to device
        images = images.to(device)

        # Because your model’s forward in eval mode returns
        #   [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
        # for the entire batch, let's call it:
        scores_batch, class_idxs_batch, transformed_anchors_batch = model(images)

        # Depending on your code, these might be flattened across the batch
        # or returned as separate lists for each image. The snippet from the
        # question suggests a single, flattened array. We'll assume batch_size == 1
        # for simplicity. If you want multi-image batch, you need logic to unflatten
        # them per image. For now, let's keep it simple.

        # If you do have batch_size=1, then scores_batch is shape [N], etc.
        # We threshold them:
        keep_idxs = scores_batch > score_thresh
        scores = scores_batch[keep_idxs]
        classes = class_idxs_batch[keep_idxs]
        boxes = transformed_anchors_batch[keep_idxs]

        # Move to CPU for evaluation
        scores = scores.detach().to(cpu_device)
        classes = classes.detach().to(cpu_device)
        boxes = boxes.detach().to(cpu_device)

        # We also get image_id from the target:
        if len(targets) == 1 and "image_id" in targets[0]:
            image_id = targets[0]["image_id"].item()
        else:
            # fallback
            image_id = -1

        # Build the dictionary to feed coco_evaluator
        # Format: {image_id: { "boxes": Tensor, "scores": Tensor, "labels": Tensor } }
        res = {}
        res[image_id] = {
            "boxes": boxes,
            "scores": scores,
            "labels": classes,
        }

        coco_evaluator.update(res)

        # Optionally store predictions & GT in JSON
        if json_path is not None:
            # We'll retrieve the original file name from COCO
            img_info = data_loader.dataset.coco.loadImgs(image_id)[0]
            file_name = img_info["file_name"]
            full_image_path = os.path.join(dataset_root, file_name)

            # Ground truth from targets
            if "boxes" in targets[0] and "labels" in targets[0]:
                gt_boxes = targets[0]["boxes"].tolist()
                gt_labels = targets[0]["labels"].tolist()
            else:
                gt_boxes = []
                gt_labels = []

            all_results[f"epoch_{epoch}"][file_name] = {
                "image_id": image_id,
                "file_name": file_name,
                "image_path": full_image_path,
                "predicted_bboxes": boxes.tolist(),
                "predicted_scores": scores.tolist(),
                "predicted_labels": classes.tolist(),
                "gt_bboxes": gt_boxes,
                "gt_labels": gt_labels,
            }

    # Final aggregator steps
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    # Save JSON if requested
    if json_path is not None:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = {}
        else:
            existing_results = {}

        existing_results.update(all_results)
        with open(json_path, "w") as f:
            json.dump(existing_results, f, indent=2)

        print(f"[INFO] Saved detection results to {json_path}")

    return coco_evaluator


def compute_validation_loss(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Computes average validation loss by temporarily setting model to training mode
    so it returns (classification_loss, regression_loss).
    """
    was_training = model.training
    model.train()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)

            formatted_targets = []
            for t in targets:
                boxes = t["boxes"].to(device)
                labels = t["labels"].to(device)
                if boxes.shape[0] == 0:
                    annotation_tensor = (
                        torch.zeros((1, 5), dtype=torch.float32, device=device) - 1
                    )
                else:
                    annotation_tensor = torch.cat([boxes, labels.unsqueeze(1)], dim=1)

                if annotation_tensor.shape[0] == 0:
                    annotation_tensor = (
                        torch.zeros((1, 5), dtype=torch.float32, device=device) - 1
                    )

                formatted_targets.append(annotation_tensor)

            classification_loss, regression_loss = model(images, formatted_targets)
            batch_loss = classification_loss + regression_loss
            total_loss += batch_loss.item()
            num_batches += 1

    if not was_training:
        model.eval()

    if num_batches == 0:
        return 0.0
    return total_loss / num_batches
