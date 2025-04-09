from typing import Dict, Any, Optional, List, Union
from torch import nn
import yaml
import os
import logging
from pathlib import Path
import time

from ICA_Detection.external.ultralytics.ultralytics import YOLO
from ICA_Detection.external.ultralytics.ultralytics.models.yolo.detect.train import (
    DetectionTrainer,
)
from ICA_Detection.external.ultralytics.ultralytics.utils import callbacks

from ICA_Detection.tasks.mga_yolo.models.hooks import HookManager
from ICA_Detection.tasks.mga_yolo.cfg.defaults import MaskGuidedAttentionConfig

# Configure module logger - matches the hierarchy in hooks.py
logger = logging.getLogger("mga_yolo.trainer")


class MaskGuidedTrainer:
    """
    Implements Mask-Guided Attention training for object detection models.

    This trainer applies segmentation masks to feature maps to guide
    the model's attention to relevant image regions during training.
    It integrates with the YOLO detection framework and adds custom
    processing to apply binary masks during forward passes.
    """

    def __init__(self, config: MaskGuidedAttentionConfig) -> None:
        """
        Initialize the Mask-Guided Attention trainer.

        Args:
            config: Configuration object for MGA training
        """
        # Setup logger first
        self._setup_logging()

        # Print distinctive ASCII art banner to clearly mark MGA usage
        self._print_mga_banner()

        self.config = config
        self.model = YOLO(config.model_cfg)
        self.masks_folder = config.masks_folder
        self.epochs = config.epochs
        self.imgsz = config.imgsz
        self.current_batch_paths: List[str] = []
        self.mga_active = True  # Flag to indicate MGA is active

        # Track mask application statistics
        self.mask_stats = {
            "total_batches": 0,
            "start_time": time.time(),
        }

        logger.info("[Trainer]: Mask-Guided Attention YOLO trainer initialized")

        # Load data configuration to get dataset structure
        with open(config.data_yaml, "r") as f:
            self.data_dict = yaml.safe_load(f)

        # Validate configuration
        if config.target_layers is None:
            raise ValueError("Target layers must be specified in the config.")

        # Initialize hook manager for handling feature map modifications
        logger.info(
            f"Calling HookManager with parameters:\n - Mask folder {config.masks_folder}\n - Target layers {config.target_layers}"
        )
        self.hook_manager = HookManager(
            masks_folder=config.masks_folder,
            target_layers=config.target_layers,
            get_image_path_fn=self._get_current_image_path,
        )

        # Log configuration details
        logger.info(
            f"[Trainer]: Trainer initialized with masks folder: {config.masks_folder}"
        )
        self._log_mask_information()

        # Log distinctive message confirming MGA setup
        for layer in config.target_layers:
            logger.info(f"[Trainer]: Feature modification registered for layer {layer}")

    def _setup_logging(self) -> None:
        """Configure logging for the trainer - matches hooks.py setup."""
        # Make sure root logger is configured
        if not logging.root.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler("mga_training.log"),
                ],
            )

        # Configure our specific logger properly
        if not logger.handlers:
            # Let it propagate to root logger
            logger.propagate = True
            logger.setLevel(logging.INFO)

    def _print_mga_banner(self) -> None:
        """Print a distinctive banner to clearly mark [Trainer] usage."""
        banner = """
        ╔═══════════════════════════════════════════════╗
        ║                                               ║
        ║    [Trainer]: Mask-Guided Attention YOLO       ║
        ║                                               ║
        ╚═══════════════════════════════════════════════╝
        """
        logger.info(banner)
        print(banner)

    def _log_mask_information(self) -> None:
        """Log information about available masks for debugging."""
        try:
            mask_files = os.listdir(self.masks_folder)
            logger.info(
                f"[Trainer]: Found {len(mask_files)} mask files in {self.masks_folder}"
            )
            if len(mask_files) > 0:
                sample_masks = mask_files[: min(5, len(mask_files))]
                logger.info(f"[Trainer]: Sample mask filenames: {sample_masks}")

                # Add distinctive logger to show mask format
                if mask_files:
                    first_mask = os.path.join(self.masks_folder, mask_files[0])
                    mask_size = os.path.getsize(first_mask)
                    logger.info(
                        f"[Trainer]: Example mask '{mask_files[0]}' has size {mask_size} bytes"
                    )
        except Exception as e:
            logger.error(f"[Trainer]: Error accessing mask folder: {e}")

    def _get_current_image_path(self, batch_idx: int) -> Optional[str]:
        """
        Get current image path from the trainer's batch.

        Args:
            batch_idx: Index of the image in the current batch

        Returns:
            Path to the image file or None if not found
        """
        if hasattr(self, "current_batch_paths") and batch_idx < len(
            self.current_batch_paths
        ):
            path = self.current_batch_paths[batch_idx]
            return path
        return None

    def _log_mga_statistics(self, batch_count: int) -> None:
        """
        Log distinctive statistics about MGA processing.

        Args:
            batch_count: Current batch count
        """
        elapsed_time = time.time() - self.mask_stats["start_time"]
        logger.info(f"[Trainer] STATS [Batch {batch_count}]:")
        logger.info(f"  - Runtime: {elapsed_time:.2f} seconds")
        logger.info(f"  - Target layers: {', '.join(self.config.target_layers)}")  # type: ignore
        logger.info(f"  - MGA active: {self.mga_active}")

    def train(self) -> YOLO:
        """
        Run the training with Mask-Guided Attention.

        This method:
        1. Registers hooks to the model
        2. Sets up a custom trainer to handle batch information
        3. Runs the YOLO training process with MGA hooks active

        Returns:
            Trained YOLO model
        """
        # Prepare the model with MGA hooks
        hook_manager = self.hook_manager
        masks_folder = self.masks_folder

        # Store reference to self for the custom trainer
        mga_trainer = self

        # Define custom trainer that will handle batch information
        class MGADetectionTrainer(DetectionTrainer):
            """
            Custom YOLO trainer that injects MGA processing during training.

            This trainer overrides methods to capture image paths from batches
            and pass them to the hook manager for mask application.
            """

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                """Initialize the MGA detection trainer."""
                super().__init__(*args, **kwargs)
                self.current_batch_paths: List[str] = []
                self.mga_trainer: Optional[MaskGuidedTrainer] = None
                self.batch_count: int = 0
                logger.info("[Trainer]: Custom MGA detection trainer initialized")
                self.add_callback("on_train_epoch_start", self._register_mga_hooks)

            def _register_mga_hooks(self, trainer):
                """Register MGA hooks after model is fully setup."""
                logger.info("[Hook Callback]: Registering MGA hooks via callback")
                hook_manager.register_hooks(self.model)

            def _do_train(self, world_size: int = 1) -> Dict[str, Any]:
                """
                Run the training process with MGA hooks.

                Args:
                    world_size: Number of GPUs for distributed training

                Returns:
                    Training results dictionary
                """
                self.mga_trainer = mga_trainer
                logger.info(f"[Trainer]: Starting MGA training with {world_size} GPUs")
                logger.info(
                    f"[Trainer]: Feature modification active on {len(self.mga_trainer.config.target_layers)} layers"  # type: ignore
                )
                return super()._do_train(world_size)

            def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
                """
                Preprocess batch and handle MGA integration.

                This method extracts image paths from the batch and
                updates the hook manager with these paths for mask matching.

                Args:
                    batch: Dictionary containing batch data

                Returns:
                    Processed batch dictionary
                """
                # Store image paths before preprocessing
                if "im_file" in batch:
                    self.current_batch_paths = batch["im_file"]
                    hook_manager.set_batch_paths(self.current_batch_paths)

                    # Update MGA trainer and hook manager with paths
                    if hasattr(self, "mga_trainer") and self.mga_trainer is not None:
                        # Increment batch count
                        self.batch_count += 1
                        self.mga_trainer.mask_stats["total_batches"] = self.batch_count

                        # Periodically log progress with distinctive MGA statistics
                        if self.batch_count % 250 == 0:
                            logger.info(
                                f"[Trainer]: Processed {self.batch_count} batches"
                            )
                            self.mga_trainer._log_mga_statistics(self.batch_count)
                    else:
                        logger.warning("[Trainer]: No mga_trainer attribute found!")

                # Call the original preprocessing
                result = super().preprocess_batch(batch)
                return result

        model = self.model
        # Log training configuration with distinctive MGA markers
        logger.info(
            f"[Trainer]: Starting training with Mask-Guided Attention for {self.epochs} epochs"
        )
        logger.info(f"[Trainer]: Masks folder: {self.masks_folder}")
        logger.info(f"[Trainer]: Image size: {self.imgsz}")
        # Start training
        results = model.train(  # type: ignore
            data=self.config.data_yaml,
            epochs=self.epochs,
            imgsz=self.imgsz,
            project=self.config.project_dir,
            name=self.config.experiment_name,
            device=self.config.device,
            batch=self.config.batch,
            trainer=MGADetectionTrainer,
            **self.config.augmentation_config,
        )

        # Log completion with distinctive MGA information
        training_time = time.time() - self.mask_stats["start_time"]
        logger.info(f"[Trainer]: Training complete in {training_time:.2f} seconds!")
        logger.info(
            f"[Trainer]: Processed {self.mask_stats['total_batches']} total batches"
        )

        return model
