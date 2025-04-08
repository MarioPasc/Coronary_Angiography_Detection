import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Union, Any

from ultralytics import YOLO
from ICA_Detection.tasks.mga_yolo.models.hooks import HookManager


def mask_guided_inference(
    model_path: str,
    image_path: str,
    mask_path: str,
    target_layers: List[str] = None,
    alpha: float = -1,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    device: str = None,
    visualize: bool = True,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform inference using Mask-Guided Attention YOLO model on a single image.

    Args:
        model_path: Path to the trained YOLO model (.pt file)
        image_path: Path to the input image for inference
        mask_path: Path to the binary segmentation mask
        target_layers: List of model layers to apply mask-guided attention to
                     (if None, uses default layers for YOLOv8)
        alpha: Weight for skip connection (-1 for residual, 0-1 for weighted combination)
        conf_thres: Confidence threshold for detections
        iou_thres: IoU threshold for non-maximum suppression
        device: Device to run inference on ('cpu', 'cuda', 'cuda:0', etc.)
        visualize: Whether to generate visualization of detections
        save_path: Path to save visualization results (if None, doesn't save)

    Returns:
        Dictionary containing detection results and visualization
    """
    # Set default target layers if not provided
    if target_layers is None:
        # Default layers where MGA is applied (typically feature pyramid layers in YOLOv8)
        target_layers = [
            "model.10",  # P3 (detection layer 1)
            "model.13",  # P4 (detection layer 2)
            "model.14",  # P5 (detection layer 3)
        ]

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create mask folder containing just this mask
    tmp_mask_folder = os.path.join(os.path.dirname(mask_path), "tmp_masks")
    os.makedirs(tmp_mask_folder, exist_ok=True)

    # Copy mask to the temporary folder with basename matching the image
    img_basename = Path(image_path).stem
    mask_basename = Path(mask_path).stem
    tmp_mask_path = os.path.join(tmp_mask_folder, f"{img_basename}.png")

    # If mask has a different name, copy it with the right name
    if mask_basename != img_basename:
        import shutil

        shutil.copy(mask_path, tmp_mask_path)
    else:
        tmp_mask_path = mask_path  # No need to copy

    # Function to get image path from batch index (required by HookManager)
    def get_image_path(batch_idx: int) -> str:
        if batch_idx == 0:  # Only handling one image
            return image_path
        return None

    # Load model
    model = YOLO(model_path)

    # Setup hook manager for mask-guided attention
    hook_manager = HookManager(
        masks_folder=os.path.dirname(tmp_mask_path),
        target_layers=target_layers,
        get_image_path_fn=get_image_path,
        alpha=alpha,
    )

    # Register hooks to apply masks during inference
    model_with_hooks = hook_manager.register_hooks(model)

    # Set current batch paths for the hook manager
    hook_manager.set_batch_paths([image_path])

    # Run inference
    results = model_with_hooks.predict(
        source=image_path, conf=conf_thres, iou=iou_thres, device=device, verbose=False
    )

    # Clean up temporary files
    if tmp_mask_path != mask_path and os.path.exists(tmp_mask_path):
        os.remove(tmp_mask_path)
    if os.path.exists(tmp_mask_folder) and not os.listdir(tmp_mask_folder):
        os.rmdir(tmp_mask_folder)

    # Clear hooks to avoid memory leaks
    hook_manager.clear_hooks()

    # Process results
    result = results[0]  # Get first (and only) result
    output = {
        "boxes": result.boxes.data.cpu().numpy(),  # Boxes (x1, y1, x2, y2, conf, cls)
        "orig_shape": result.orig_shape,  # Original image shape
        "names": result.names,  # Class names dictionary
    }

    # Generate visualization if requested
    if visualize:
        # Get the plotted image with boxes
        plotted_img = result.plot()
        output["visualization"] = plotted_img

        # Save visualization if save_path is provided
        if save_path:
            Image.fromarray(plotted_img).save(save_path)

    return output


# Usage example
def visualize_mask_guided_detection():
    """Example of how to use the mask_guided_inference function."""
    # Paths to model, image and mask
    model_path = "path/to/trained/model.pt"
    image_path = "path/to/image.jpg"
    mask_path = "path/to/mask.png"
    save_path = "results/mga_detection.jpg"

    # Run inference
    results = mask_guided_inference(
        model_path=model_path,
        image_path=image_path,
        mask_path=mask_path,
        alpha=-1,  # Use residual connection
        conf_thres=0.25,
        iou_thres=0.45,
        device="cuda",
        visualize=True,
        save_path=save_path,
    )

    # Access detection results
    boxes = results["boxes"]
    print(f"Detected {len(boxes)} objects")

    # Display visualization using PIL
    if "visualization" in results:
        Image.fromarray(results["visualization"]).show()

    return results
