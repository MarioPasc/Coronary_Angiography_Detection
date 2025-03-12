import math
import sys
import time
import os
import json

from typing import Dict, Any

import torch
import torchvision.models.detection.mask_rcnn
from . import utils
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset

DEBUG: bool = False


def train_one_epoch(
    model, optimizer, data_loader, device, epoch, print_freq, scaler=None
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)

        formatted_targets = []
        for t in targets:
            if t.shape[0] == 0:  # Handle empty annotations
                annotation_tensor = (
                    torch.zeros((1, 5), dtype=torch.float32, device=device) - 1
                )
            else:
                boxes = t[:, :4]  # Extract bounding boxes
                labels = t[:, 4].view(-1, 1)  # Extract labels and ensure correct shape
                annotation_tensor = torch.cat([boxes, labels], dim=1).to(device)
            # Handle cases where there are no annotations
            if annotation_tensor.shape[0] == 0:
                annotation_tensor = (
                    torch.zeros((1, 5), dtype=torch.float32, device=device) - 1
                )

            formatted_targets.append(annotation_tensor)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            # Model returns a tuple of (classification_loss, regression_loss)
            classification_loss, regression_loss = model(images, formatted_targets)
            # Sum the losses
            losses = classification_loss + regression_loss

            # Create a loss dict for metric logging
            loss_dict = {
                "classification": classification_loss,
                "regression": regression_loss,
            }

            # For distributed training (keeping your existing code structure)
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
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
def evaluate(model, data_loader, device, epoch, json_path=None):
    """
    If save_json=True, we collect predictions + ground-truth in all_results
    and save them to json_path at the end.
    """
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Evaluation on epoch {epoch}:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    # Only accumulate results if we plan to save them.
    all_results = {}

    all_results[f"epoch_{epoch}"] = {}

    dataset_root = data_loader.dataset.root  # Where images are stored

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = [img.to(device) for img in images]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()

        # Switch model to eval mode for inference
        model.eval()
        with torch.no_grad():
            # During evaluation, we need predictions, not losses
            # We need to modify how we call the model or post-process its outputs
            raw_predictions = model(images)

            # Process the raw predictions into the expected format
            # This depends on your model architecture, but typically we need:
            # - boxes, scores, labels for each image
            outputs = []

            # If model returns pre-formatted predictions in eval mode:
            if isinstance(raw_predictions, list) and all(
                isinstance(pred, dict) for pred in raw_predictions
            ):
                outputs = raw_predictions
            # If model returns a tuple of tensors (e.g., from RetinaNet):
            elif isinstance(raw_predictions, tuple):
                # Assuming raw_predictions contains classification and regression outputs
                # We need to decode them into boxes, scores, and labels
                for i in range(len(images)):
                    # Process predictions for each image
                    boxes, scores, labels = process_predictions(raw_predictions, i)
                    outputs.append({"boxes": boxes, "scores": scores, "labels": labels})

        model_time = time.time() - model_time

        # Move outputs to CPU for evaluation and JSON saving
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # Prepare results for coco evaluator
        res = {}
        for tgt, out in zip(targets, outputs):
            # We have "image_id" stored in the target
            img_id_tensor = tgt["image_id"]
            if isinstance(img_id_tensor, torch.Tensor):
                img_id = img_id_tensor.item()
            else:
                img_id = img_id_tensor  # in case it's int

            res[img_id] = out

            if json_path is not None:
                # 1) Get file name from COCO
                #    (this also works if your COCO "images" contain 'file_name')
                img_info = data_loader.dataset.coco.loadImgs(img_id)[0]
                file_name = img_info["file_name"]
                # Full path
                full_image_path = os.path.join(dataset_root, file_name)

                # 2) Predictions
                predicted_bboxes = out["boxes"].tolist()  # Nx4
                predicted_scores = out["scores"].tolist()  # Nx
                predicted_labels = out["labels"].tolist()  # Nx

                # 3) Ground truth
                gt_bboxes = tgt["boxes"].tolist()  # Mx4
                gt_labels = tgt["labels"].tolist()  # Mx
                gt_area = tgt["area"].tolist()  # Mx
                gt_iscrowd = tgt["iscrowd"].tolist()  # Mx

                all_results[f"epoch_{epoch}"][f"{file_name}"] = {
                    "image_id": img_id,
                    "file_name": file_name,
                    "image_path": full_image_path,
                    "predicted_bboxes": predicted_bboxes,
                    "predicted_scores": predicted_scores,
                    "predicted_labels": predicted_labels,
                    "gt_bboxes": gt_bboxes,
                    "gt_labels": gt_labels,
                    "gt_area": gt_area,
                    "gt_iscrowd": gt_iscrowd,
                }

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        if DEBUG:
            print(f"[DEBUG] Processed batch with {len(images)} images.")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    # If requested, dump all_results to JSON
    if json_path is not None:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        # Load existing JSON if available
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                try:
                    existing_results = json.load(f)
                except json.JSONDecodeError:
                    existing_results = {}  # In case of a corrupt or empty JSON file
        else:
            existing_results = {}

        # Update with new epoch results
        existing_results.update(all_results)

        # Write back the merged results
        with open(json_path, "w") as f:
            json.dump(existing_results, f, indent=2)

        print(f"[INFO] Saved detection results to {json_path}")

    return coco_evaluator


def process_predictions(raw_predictions, image_index):
    """
    Process raw model predictions into the expected format of boxes, scores, and labels.
    This function needs to be adapted based on your specific model architecture.

    Args:
        raw_predictions: The raw output from your model
        image_index: The index of the current image in the batch

    Returns:
        boxes: Tensor of bounding boxes in [x1, y1, x2, y2] format
        scores: Tensor of confidence scores
        labels: Tensor of class labels
    """
    # This is a placeholder implementation that needs to be adapted to your model
    # For example, if you're using RetinaNet with FocalLoss:

    classification_output, regression_output = raw_predictions

    # For RetinaNet-like models, you'd typically:
    # 1. Select outputs for the current image
    # 2. Apply sigmoid to classification outputs
    # 3. Transform regression outputs to boxes
    # 4. Apply NMS

    # This is simplified and needs to be adapted to your specific architecture
    import torch.nn.functional as F

    # Get predictions for current image
    cls_out = classification_output[image_index]
    reg_out = regression_output[image_index]

    # Apply sigmoid to get scores
    scores = F.sigmoid(cls_out).max(dim=1)[0]

    # Get class labels
    labels = F.sigmoid(cls_out).max(dim=1)[1] + 1  # Adding 1 assuming background is 0

    # Here you would transform regression outputs to actual boxes
    # This is model-specific and needs implementation based on your architecture
    # For now, we'll create dummy boxes as a placeholder
    boxes = torch.zeros((len(scores), 4), device=reg_out.device)

    # Filter predictions with low confidence
    keep = scores > 0.05
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    return boxes, scores, labels


def compute_validation_loss(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Computes the average validation loss for a Torchvision detection model.

    Because the model returns losses only when in training mode,
    we temporarily set model.train() but still wrap the computation
    in a no-grad block to avoid updating weights.

    Args:
        model (torch.nn.Module): Your detection model (Faster R-CNN, SSD, etc.).
        data_loader (torch.utils.data.DataLoader): Validation data loader that yields
            (images, targets) pairs.
        device (torch.device): The device ('cpu' or 'cuda').

    Returns:
        float: The average validation loss over all batches in data_loader.
    """
    # Store the original training/eval mode so we can restore it at the end
    was_training = model.training

    # Put model into training mode so it returns the loss dict
    model.train()

    total_loss = 0.0
    num_batches = 0

    # We do not want gradient updates for validation
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]

            # Format targets similar to the train_one_epoch function
            formatted_targets = []
            for t in targets:
                if t.shape[0] == 0:  # Handle empty annotations
                    annotation_tensor = (
                        torch.zeros((1, 5), dtype=torch.float32, device=device) - 1
                    )
                else:
                    boxes = t[:, :4]  # Extract bounding boxes
                    labels = t[:, 4].view(
                        -1, 1
                    )  # Extract labels and ensure correct shape
                    annotation_tensor = torch.cat([boxes, labels], dim=1).to(device)
                # Handle cases where there are no annotations
                if annotation_tensor.shape[0] == 0:
                    annotation_tensor = (
                        torch.zeros((1, 5), dtype=torch.float32, device=device) - 1
                    )

                formatted_targets.append(annotation_tensor)

            # Get losses from model (assuming model returns tuple of (class_loss, reg_loss))
            classification_loss, regression_loss = model(images, formatted_targets)

            # Sum the losses to get total batch loss
            batch_loss = classification_loss + regression_loss

            total_loss += batch_loss.item()
            num_batches += 1

    # Restore model's original mode
    if not was_training:
        model.eval()

    if num_batches == 0:
        return 0.0
    return total_loss / num_batches
