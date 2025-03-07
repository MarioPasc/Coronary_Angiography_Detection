import math
import sys
import time
import os
import json

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
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
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
        outputs = model(images)
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
                predicted_bboxes = out["boxes"].tolist()   # Nx4
                predicted_scores = out["scores"].tolist()  # Nx
                predicted_labels = out["labels"].tolist()  # Nx

                # 3) Ground truth
                gt_bboxes = tgt["boxes"].tolist()          # Mx4
                gt_labels = tgt["labels"].tolist()         # Mx
                gt_area = tgt["area"].tolist()             # Mx
                gt_iscrowd = tgt["iscrowd"].tolist()       # Mx

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
                    "gt_iscrowd": gt_iscrowd
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
