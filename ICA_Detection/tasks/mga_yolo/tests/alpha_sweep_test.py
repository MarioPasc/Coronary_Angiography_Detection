import sys
import logging
import argparse
import os
from datetime import datetime
from typing import List, Optional, Tuple

from ICA_Detection.tasks.mga_yolo.cfg.defaults import MaskGuidedAttentionConfig
from ICA_Detection.tasks.mga_yolo.engine.trainer import MaskGuidedTrainer

import numpy as np


def setup_logging(log_file: str) -> None:
    """
    Configure logging for the MGA alpha sweep experiment.

    Args:
        log_file: Path to the log file
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run MGA training with different alpha values"
    )

    parser.add_argument(
        "--model", type=str, default="yolov8n.pt", help="Path to model weights"
    )
    parser.add_argument(
        "--data-yaml",
        type=str,
        default="/home/mariopasc/Python/Datasets/COMBINED/YOLO_MGA/detection/yolo_ica_detection.yaml",
        help="Path to data YAML file",
    )
    parser.add_argument(
        "--masks-folder",
        type=str,
        default="/home/mariopasc/Python/Datasets/COMBINED/YOLO_MGA/masks",
        help="Path to folder containing masks",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "--imgsz", type=int, default=512, help="Image size for training"
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default="/home/mariopasc/Python/Datasets/COMBINED/detection/runs/train/alpha_sweep",
        help="Base directory for saving results",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=np.linspace(0.5, 1, 15),
        help="Alpha values to test",
    )
    parser.add_argument(
        "--target-layers",
        type=str,
        nargs="+",
        default=["model.15", "model.18", "model.21"],
        help="Target layers for MGA",
    )

    return parser.parse_args()


def get_experiment_directory(base_dir: str) -> str:
    """
    Create a timestamped experiment directory.

    Args:
        base_dir: Base directory for experiments

    Returns:
        Path to the new experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"alpha_sweep_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def run_mga_alpha_sweep(
    model_path: str,
    data_yaml: str,
    masks_folder: str,
    epochs: int,
    imgsz: int,
    project_dir: str,
    alpha_values: List[float],
    target_layers: List[str],
) -> List[Tuple[float, Optional[str]]]:
    """
    Run MGA training with different alpha values.

    Args:
        model_path: Path to model weights
        data_yaml: Path to data YAML file
        masks_folder: Path to masks folder
        epochs: Number of training epochs
        imgsz: Image size
        project_dir: Base directory for results
        alpha_values: List of alpha values to test
        target_layers: List of target layers for MGA

    Returns:
        List of tuples containing (alpha, checkpoint_path) for each run
    """
    results = []

    # Create experiment directory
    exp_dir = get_experiment_directory(project_dir)
    logging.info(f"Created experiment directory at {exp_dir}")

    # Setup logging for this experiment
    log_file = os.path.join(exp_dir, "alpha_sweep.log")
    setup_logging(log_file)

    # Log experiment setup
    logging.info(f"Starting MGA alpha sweep with values: {alpha_values}")
    logging.info(f"Model: {model_path}")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Target layers: {target_layers}")

    # Iterate through alpha values
    for alpha in alpha_values:
        try:
            logging.info(f"\n\n{'=' * 50}")
            logging.info(f"Starting training with alpha = {alpha}")
            logging.info(f"{'=' * 50}\n")

            # Create specific experiment name including alpha value
            experiment_name = f"mga_yolo_alpha_{alpha:.2f}"

            # Create configuration
            config = MaskGuidedAttentionConfig(
                model_cfg=model_path,
                data_yaml=data_yaml,
                masks_folder=masks_folder,
                epochs=epochs,
                imgsz=imgsz,
                project_dir=exp_dir,
                experiment_name=experiment_name,
                alpha=alpha,
                target_layers=target_layers,
            )

            # Initialize trainer
            trainer = MaskGuidedTrainer(config)

            # Run training
            trained_model = trainer.train()

            # Store result
            checkpoint_path = getattr(trained_model, "ckpt_path", None)
            results.append((alpha, checkpoint_path))

            logging.info(f"Completed training with alpha = {alpha}")
            if checkpoint_path:
                logging.info(f"Model saved to {checkpoint_path}")

        except Exception as e:
            logging.error(f"Error running training with alpha={alpha}: {e}")
            results.append((alpha, None))

    return results


def main() -> None:
    """Main entry point for alpha sweep experiment."""
    args = parse_arguments()

    # Run alpha sweep
    results = run_mga_alpha_sweep(
        model_path=args.model,
        data_yaml=args.data_yaml,
        masks_folder=args.masks_folder,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project_dir=args.project_dir,
        alpha_values=args.alphas,
        target_layers=args.target_layers,
    )

    # Print summary results
    print("\nAlpha Sweep Results:")
    print("-" * 60)
    print("Alpha\tCheckpoint Path")
    print("-" * 60)
    for alpha, path in results:
        status = "Success" if path else "Failed"
        print(f"{alpha:.2f}\t{status}")
    print("-" * 60)


if __name__ == "__main__":
    main()
