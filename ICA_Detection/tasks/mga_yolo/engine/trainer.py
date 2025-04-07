from typing import Dict, Any, Optional, List, Union
import torch
import yaml
import os
import logging
from pathlib import Path
from PIL import Image
import traceback

from ICA_Detection.external.ultralytics.ultralytics import YOLO
from ICA_Detection.external.ultralytics.ultralytics.models.yolo.detect.train import (
    DetectionTrainer,
)
from ICA_Detection.external.ultralytics.ultralytics.utils import callbacks

from ICA_Detection.tasks.mga_yolo.models.hooks import HookManager
from ICA_Detection.tasks.mga_yolo.cfg.defaults import MaskGuidedAttentionConfig
from ICA_Detection.tasks.mga_yolo.utils.visualize import create_mga_visualization


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
        self.config = config
        self.model = YOLO(config.model_cfg)
        self.masks_folder = config.masks_folder
        self.epochs = config.epochs
        self.imgsz = config.imgsz
        self.visualize_interval = config.visualize_interval
        self.current_batch_paths: List[str] = []

        # Load data configuration to get dataset structure
        with open(config.data_yaml, "r") as f:
            self.data_dict = yaml.safe_load(f)

        # Validate configuration
        if config.target_layers is None:
            raise ValueError("Target layers must be specified in the config.")

        # Initialize hook manager for handling feature map modifications
        self.hook_manager = HookManager(
            masks_folder=config.masks_folder,
            target_layers=config.target_layers,
            get_image_path_fn=self._get_current_image_path,
        )

        # Log configuration details
        logging.info(
            f"MGA Trainer initialized with masks folder: {config.masks_folder}"
        )
        self._log_mask_information()

    def _log_mask_information(self) -> None:
        """Log information about available masks for debugging."""
        try:
            mask_files = os.listdir(self.masks_folder)
            logging.info(f"Found {len(mask_files)} mask files in {self.masks_folder}")
            if len(mask_files) > 0:
                sample_masks = mask_files[: min(5, len(mask_files))]
                logging.info(f"Sample mask filenames: {sample_masks}")
        except Exception as e:
            logging.error(f"Error accessing mask folder: {e}")

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

    def visualize_mga_process(self, batch_idx: int) -> None:
        """
        Create a visualization of the MGA process for the current batch.

        This method creates a grid visualization showing:
        - Original input image
        - Segmentation mask
        - Feature maps before masking
        - Resized masks for each feature level
        - Feature maps after masking

        Args:
            batch_idx: Current batch index for naming the visualization file
        """
        # Get visualization data from hook manager
        visualization_data = self.hook_manager.get_visualization_data()

        # Create visualization directory
        save_dir = os.path.join(
            self.config.project_dir, self.config.experiment_name, "visualizations"
        )
        os.makedirs(save_dir, exist_ok=True)

        # Create and save visualization
        try:
            create_mga_visualization(visualization_data, batch_idx, save_dir)
            logging.info(f"Visualization saved for batch {batch_idx}")
        except Exception as e:
            logging.error(f"Error creating visualization: {e}")
            logging.error(traceback.format_exc())

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
        model = self.hook_manager.register_hooks(self.model)

        # Store reference to self for the custom trainer
        mga_trainer = self

        # Define custom trainer that will handle batch information
        class MGADetectionTrainer(DetectionTrainer):
            """
            Custom YOLO trainer that injects MGA processing during training.

            This trainer overrides methods to:
            - Capture image paths from batches
            - Trigger mask application via hooks
            - Generate visualizations at specified intervals
            """

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                """Initialize the MGA detection trainer."""
                super().__init__(*args, **kwargs)
                self.current_batch_paths: List[str] = []
                self.mga_trainer: Optional[MaskGuidedTrainer] = None
                self.batch_count: int = 0
                logging.info("MGADetectionTrainer initialized")

            def _do_train(self, world_size: int = 1) -> Dict[str, Any]:
                """
                Run the training process with MGA hooks.

                Args:
                    world_size: Number of GPUs for distributed training

                Returns:
                    Training results dictionary
                """
                self.mga_trainer = mga_trainer
                logging.info(f"Starting MGA training with {world_size} GPUs")
                return super()._do_train(world_size)

            def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
                """
                Preprocess batch and handle MGA integration.

                This method:
                1. Extracts image paths from the batch
                2. Updates the MGA trainer and hook manager with these paths
                3. Determines if the current batch should be visualized
                4. Triggers visualization after batch processing if needed

                Args:
                    batch: Dictionary containing batch data

                Returns:
                    Processed batch dictionary
                """
                # Store image paths before preprocessing
                if "im_file" in batch:
                    self.current_batch_paths = batch["im_file"]

                    # Update MGA trainer and hook manager with paths
                    if hasattr(self, "mga_trainer") and self.mga_trainer is not None:
                        # Copy paths to MGA trainer
                        self.mga_trainer.current_batch_paths = (
                            self.current_batch_paths.copy()
                        )

                        # Update hook manager with paths
                        self.mga_trainer.hook_manager.set_batch_paths(
                            self.current_batch_paths
                        )

                        logging.debug(
                            f"Set batch paths with {len(self.current_batch_paths)} items"
                        )

                        # Log sample paths for debugging
                        if len(self.current_batch_paths) > 0:
                            sample_paths = [
                                Path(p).name for p in self.current_batch_paths[:2]
                            ]
                            logging.debug(f"Sample paths: {sample_paths}")

                        # Increment batch count
                        self.batch_count += 1

                        # Check if we should visualize this batch
                        should_visualize = (
                            self.batch_count % self.mga_trainer.visualize_interval == 0
                        )

                        # Update visualization state in hook manager
                        self.mga_trainer.hook_manager.set_visualization_state(
                            should_visualize, self.batch_count
                        )

                        if should_visualize:
                            logging.info(f"Will visualize batch {self.batch_count}")
                    else:
                        logging.warning("No mga_trainer attribute found!")

                # Call the original preprocessing
                result = super().preprocess_batch(batch)

                # After batch is processed, create visualization if needed
                if (
                    hasattr(self, "mga_trainer")
                    and self.mga_trainer is not None
                    and self.mga_trainer.hook_manager.visualize_current_batch
                ):
                    try:
                        self.mga_trainer.visualize_mga_process(self.batch_count)
                        logging.info("Visualization completed successfully")
                    except Exception as e:
                        logging.error(f"Error creating visualization: {e}")
                        logging.error(traceback.format_exc())

                    # Reset visualization flag
                    self.mga_trainer.hook_manager.visualize_current_batch = False

                return result

        # Log training configuration
        logging.info(
            f"Starting training with Mask-Guided Attention for {self.epochs} epochs"
        )
        logging.info(f"Masks folder: {self.masks_folder}")
        logging.info(
            f"Will generate visualizations every {self.visualize_interval} batches"
        )
        logging.info(f"Image size: {self.imgsz}")
        logging.info(f"Target layers for MGA: {self.config.target_layers}")

        # Start training
        results = model.train(  # type: ignore
            data=self.config.data_yaml,
            epochs=self.epochs,
            imgsz=self.imgsz,
            project=self.config.project_dir,
            name=self.config.experiment_name,
            device=self.config.device,
            trainer=MGADetectionTrainer,
            **self.config.augmentation_config,
        )

        # Log final statistics
        logging.info("Training complete!")
        logging.info(f"Final MGA Stats: {self.hook_manager.get_stats()}")

        return model
