from ICA_Detection.external.ultralytics.ultralytics import YOLO

#!/usr/bin/env python3
"""
YOLO Model Training Script for Coronary Angiography Detection

This script trains a YOLO model for detecting features in coronary angiography images
using the Ultralytics YOLO implementation with a custom configuration file.
"""

import os
import argparse
from typing import Dict, Any, Literal, Union, Optional

from ICA_Detection.external.ultralytics.ultralytics import YOLO


def train_model(
    config_path: str,
    weights: str = "yolov8n.pt",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "",
    project: str = "runs/train",
    name: str = "exp",
) -> Dict[str, Any]:
    """
    Train a YOLO model using the specified configuration and parameters.

    Args:
        config_path: Path to the YOLO configuration file
        weights: Path to pre-trained weights or model name
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Image size for training
        device: Device to use (cuda device or cpu)
        project: Project name for saving results
        name: Experiment name for saving results

    Returns:
        Dictionary containing training results
    """
    # Validate the config path exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading YOLO model with weights: {weights}")
    model: YOLO = YOLO(weights)

    print(f"Starting training with config: {config_path}")
    print("All augmentations disabled")
    results: Dict[str, Any] = model.train(
        data=config_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        # Disable all augmentation options
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        bgr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment=None,
        erasing=0.0,
        crop_fraction=0.0,
    )

    print(f"Training completed. Results saved to: {os.path.join(project, name)}")
    return results


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for YOLO training."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train YOLO model for coronary angiography detection"
    )

    config_path: str = (
        "/home/mariopasc/Python/Datasets/COMBINED/YOLO_MGA/detection/yolo_ica_detection.yaml"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=config_path,
        help="Path to YOLO configuration file",
    )
    parser.add_argument(
        "--weights", type=str, default="yolov8n.pt", help="Initial weights path"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size for training")
    parser.add_argument(
        "--img-size", type=int, default=512, help="Image size for training"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda device or cpu)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="/home/mariopasc/Python/Datasets/COMBINED/detection/runs/train",
        help="Project name",
    )
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")

    return parser.parse_args()


def main() -> None:
    """Main function to parse arguments and train the YOLO model."""
    args: argparse.Namespace = parse_arguments()

    results: Dict[str, Any] = train_model(
        config_path=args.config,
        weights=args.weights,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
