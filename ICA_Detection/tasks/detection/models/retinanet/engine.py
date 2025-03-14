import math
import sys
import os
import json

import torch
import torchvision.models.detection.mask_rcnn

from ICA_Detection.tasks.detection.utils import utils

import json
import torch
from pycocotools.cocoeval import COCOeval


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

        # All the images are stacked within the img entry of the batch
        # therefore, it is normal to have a single image tensor and a
        # single target tensor for each batch.
        # This means that the batch is a dictionary with only 3 keys, not
        # a list of dictionaries.
        images = batch['img']
        targets = batch['annot']
        scales = batch['scale']

        """
        print(f"[DEBUG] type(batch): {type(batch)}")
        print(f"[DEBUG] len(batch): {len(batch)}")
        print(f"[DEBUG] batch keys: {batch.keys()}")
        print(f"[DEBUG] images shape: {images.shape}")
        print(f"[DEBUG] targets shape: {targets.shape}")
        print(f"[DEBUG] scales: {scales}")
        """
        images = images.to(device)
        targets = targets.to(device)

        # Convert each target dict -> Nx5 for FocalLoss
        formatted_targets = []
        for t in targets:
            valid_indices = t[:, 0] != -1  # Filter out the padded annotations
            boxes = t[valid_indices, :4].to(device)
            labels = t[valid_indices, 4].to(device)
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
            classification_loss, regression_loss = model([images, formatted_targets])
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
def evaluate(model, data_loader, device, epoch, json_path=None, score_thresh=0.05):
    """
    Evaluates a detection model on a COCO dataset using pycocotools to compute mAP.
    
    In inference mode, the model's forward pass returns a list of three tensors:
      - finalScores: Tensor of detection scores.
      - finalAnchorBoxesIndexes: Tensor of predicted class indices.
      - finalAnchorBoxesCoordinates: Tensor of bounding boxes in [xmin, ymin, xmax, ymax] format.
    
    The function applies a score threshold, converts the bounding boxes to COCO format
    ([x, y, w, h]), adjusts for image scaling, accumulates predictions into a JSON file,
    and runs the COCO evaluation.
    
    Parameters:
        model (torch.nn.Module): The detection model.
        data_loader (torch.utils.data.DataLoader): DataLoader yielding evaluation samples.
        device (torch.device): Device on which to run the model.
        epoch (int): Current epoch (for logging and result organization).
        json_path (str, optional): Path to save the prediction JSON file.
        score_thresh (float): Confidence threshold for filtering detections.
    
    Returns:
        coco_eval (COCOeval): The COCO evaluation object with evaluation metrics.
    """
    model.eval()  # set model to evaluation mode
    cpu_device = torch.device("cpu")
    results = []   # list to accumulate predictions in COCO format

    # Iterate over the evaluation dataset; here we assume batch size = 1.
    for idx, data in enumerate(data_loader):
        # Extract image and scale factor from the collated batch.
        # Note: collater returns a list for 'scale', so we take the first element.
        image = data['img'].to(device)
        scale = data.get('scale', [1.0])[0]
        
        # Run inference without gradient computation.
        with torch.no_grad():
            outputs = model(image)
        
        # Process the model's output.
        # The forward pass is expected to return a list of three tensors.
        if isinstance(outputs, list) and len(outputs) == 3 and isinstance(outputs[0], torch.Tensor):
            scores, labels, boxes = outputs
        else:
            # Fallback if a different output format is returned.
            output = outputs[0]
            scores = output.get('scores', torch.Tensor([]))
            labels = output.get('labels', torch.Tensor([]))
            boxes = output.get('boxes', torch.Tensor([]))
        
        # Move predictions to CPU.
        scores = scores.cpu()
        labels = labels.cpu()
        boxes = boxes.cpu()
        
        # Apply score threshold filtering.
        keep = scores > score_thresh
        scores = scores[keep]
        labels = labels[keep]
        boxes  = boxes[keep]
        
        # Obtain the image_id from the dataset.
        # Here we assume the order in the DataLoader corresponds to dataset.image_ids.
        image_id = data_loader.dataset.image_ids[idx]
        
        # Skip this image if no detections remain.
        if boxes.shape[0] == 0:
            continue
        
        # Convert boxes from [xmin, ymin, xmax, ymax] to [x, y, w, h].
        boxes_converted = boxes.clone()
        boxes_converted[:, 2] = boxes_converted[:, 2] - boxes_converted[:, 0]  # width
        boxes_converted[:, 3] = boxes_converted[:, 3] - boxes_converted[:, 1]  # height

        # Adjust bounding boxes for the image scale.
        if isinstance(scale, (int, float)):
            boxes_converted = boxes_converted / scale
        else:
            boxes_converted = boxes_converted / scale
        
        # Map predicted labels to COCO category IDs using the dataset's mapping function.
        if hasattr(data_loader.dataset, 'label_to_coco_label'):
            # Use the method provided by the dataset.
            map_label = lambda lbl: data_loader.dataset.label_to_coco_label(lbl)
        else:
            map_label = lambda lbl: int(lbl)
        
        # Build a COCO-formatted prediction for each detection.
        for j in range(boxes_converted.shape[0]):
            result = {
                'image_id': image_id,
                'category_id': map_label(int(labels[j])),
                'score': float(scores[j]),
                'bbox': boxes_converted[j].tolist(),  # in [x, y, w, h] format
            }
            results.append(result)
        
        print(f'Processed {idx + 1}/{len(data_loader)}', end='\r')
    
    # Check if results are empty
    if not results:
        print("No predictions were made.")
        return
    
    # Determine JSON file path for saving results.
    if json_path is None:
        json_path = f'{data_loader.dataset.set_name}_bbox_results.json' if hasattr(data_loader.dataset, 'set_name') else 'bbox_results.json'
    
    # Save predictions to the JSON file.
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Load ground truth annotations via the COCO API.
    if hasattr(data_loader.dataset, 'coco'):
        coco_true = data_loader.dataset.coco
    else:
        print("Dataset does not provide COCO ground truth information.")
        return
    
    coco_pred = coco_true.loadRes(json_path)
    
    # Run the COCO evaluation.
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    # Restrict evaluation to the processed images.
    coco_eval.params.imgIds = [data_loader.dataset.image_ids[i] for i in range(len(data_loader.dataset))]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Set the model back to training mode.
    model.train()
    return coco_eval



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
        for batch in data_loader:
            images = batch['img']
            targets = batch['annot']

            images = images.to(device)

            formatted_targets = []
            for t in targets:
                valid_indices = t[:, 0] != -1  # Filter out the padded annotations
                boxes = t[valid_indices, :4].to(device)
                labels = t[valid_indices, 4].to(device)
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

            classification_loss, regression_loss = model([images, formatted_targets])
            batch_loss = classification_loss + regression_loss
            total_loss += batch_loss.item()
            num_batches += 1

    if not was_training:
        model.eval()

    if num_batches == 0:
        return 0.0
    return total_loss / num_batches
