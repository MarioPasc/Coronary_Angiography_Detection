from ICA_Detection.external.ultralytics.ultralytics import YOLO
from ICA_Detection.external.ultralytics.ultralytics.engine.trainer import BaseTrainer
from ICA_Detection.external.ultralytics.ultralytics.utils import callbacks, DEFAULT_CFG
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import yaml
import sys


import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("mga_training.log")],
)


class MaskGuidedTrainer:
    def __init__(
        self,
        model_cfg,
        data_yaml,
        masks_folder,
        epochs=100,
        imgsz=640,
        visualize_interval=100,
    ):
        """
        Initialize the Mask-Guided Attention trainer

        Args:
            model_cfg: Path to model config or pre-trained weights
            data_yaml: Path to data YAML file
            masks_folder: Path to folder containing masks
            epochs: Number of training epochs
            imgsz: Input image size
            visualize_interval: Interval (in batches) for creating visualizations
        """
        self.model = YOLO(model_cfg)
        self.data_yaml = data_yaml
        self.masks_folder = masks_folder
        self.epochs = epochs
        self.imgsz = imgsz
        self.visualize_interval = visualize_interval

        # Load data configuration to get dataset structure
        with open(data_yaml, "r") as f:
            self.data_dict = yaml.safe_load(f)

        # Original hooks storage to prevent memory leaks
        self.hooks = []

        # Storage for visualization data
        self.visualization_data = {
            "original_input": None,
            "mask": None,
            "feature_maps": {},  # Will store feature maps before masking
            "downsized_masks": {},  # Will store resized masks for each feature map
            "masked_feature_maps": {},  # Will store feature maps after masking
            "predictions": None,  # Will store model predictions (optional)
        }

        # Statistics for tracking MGA application
        self.stats = {
            "total_images_processed": 0,
            "images_with_masks": 0,
            "images_without_masks": 0,
            "masks_applied": 0,
            "masks_failed": 0,
        }

        logging.info(f"MGA Trainer initialized with masks folder: {masks_folder}")

        # Log available masks
        mask_files = os.listdir(masks_folder)
        logging.info(f"Found {len(mask_files)} mask files in {masks_folder}")
        if len(mask_files) > 0:
            logging.info(f"Sample mask filenames: {mask_files[:5]}")

    def get_capture_hook(self, layer_name):
        """Create a hook function that captures feature maps before masking"""

        def hook(module, input_feat, output):
            # Only capture for visualization when needed
            if not hasattr(self, "current_batch_idx") or not hasattr(
                self, "visualize_current_batch"
            ):
                return output

            if self.visualize_current_batch:
                # Store the original feature maps for visualization
                self.visualization_data["feature_maps"][layer_name] = (
                    output[0:1].detach().clone()
                )

            return output

        return hook

    def _get_current_image_path(self, batch_idx):
        """Get current image path from the trainer's batch"""
        # Add debugging to see if this method is being called
        logging.info(f"Attempting to get image path for batch index {batch_idx}")

        # This is a placeholder - the actual implementation depends on how
        # the ultralytics trainer stores image paths
        try:
            # Access the current batch info
            if hasattr(self, "current_batch_paths") and batch_idx < len(
                self.current_batch_paths
            ):
                path = self.current_batch_paths[batch_idx]
                logging.info(f"Found image path: {path}")
                return path
            else:
                if not hasattr(self, "current_batch_paths"):
                    logging.warning("current_batch_paths attribute not found!")
                elif batch_idx >= len(self.current_batch_paths):
                    logging.warning(
                        f"Batch index {batch_idx} out of range for {len(self.current_batch_paths)} paths"
                    )
                return None
        except Exception as e:
            logging.error(f"Error getting image path: {e}")
            return None

    def register_mga_hooks(self, model):
        """Register hooks for Mask-Guided Attention to specified layers"""
        # Clear any existing hooks to prevent memory leaks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        # Layer names to hook for visualization
        layer_names = ["model.15", "model.18", "model.21"]  # P3, P4, P5 features

        # Check if these layers exist
        found_layers = []
        for name, module in model.model.named_modules():
            if name in layer_names:
                found_layers.append(name)

        logging.info(f"Found target layers: {found_layers}")
        print(f"Found target layers: {found_layers}")  # Direct console output

        # Register new hooks
        for name, module in model.model.named_modules():
            if isinstance(module, torch.nn.Module) and name.startswith("model."):
                if name in layer_names:
                    print(f"Registering MGA hook for {name}")
                    # First hook to capture feature maps before masking
                    capture_hook = self.get_capture_hook(name)
                    self.hooks.append(module.register_forward_hook(capture_hook))
                    logging.info(f"Registered capture hook for {name}")

                    # Second hook to apply masks and capture masked feature maps
                    mga_hook = self.get_mga_hook(name)
                    self.hooks.append(module.register_forward_hook(mga_hook))
                    logging.info(f"Registered MGA hook for {name}")

        return model

    def get_debug_hook(self, layer_name):
        """Create a debug hook to verify hooks are being called"""

        def hook(module, input_feat, output):
            print(
                f"DEBUG: Hook for {layer_name} was called! Output shape: {output.shape}"
            )
            logging.info(
                f"DEBUG: Hook for {layer_name} was called! Output shape: {output.shape}"
            )
            return output

        return hook

    def get_mga_hook(self, layer_name=None):
        """Create a hook function that applies the mask to feature maps"""

        def hook(module, input_feat, output):
            # The batch contains multiple images
            batch_size = output.shape[0]
            modified_outputs = []

            # Log layer information
            if layer_name:
                logging.debug(
                    f"Processing layer {layer_name}, output shape: {output.shape}"
                )

            # Process each image in the batch
            for i in range(batch_size):
                # Update statistics
                self.stats["total_images_processed"] += 1

                # Capture for visualization if this is the first item in the batch
                # and we're visualizing this batch
                should_visualize = (
                    i == 0
                    and hasattr(self, "visualize_current_batch")
                    and self.visualize_current_batch
                )

                # Get current image name from trainer.batch
                try:
                    # This part is tricky and depends on how ultralytics stores paths
                    # We'll need to extract them from the current batch
                    img_path = self._get_current_image_path(i)
                    if img_path is None:
                        logging.warning(f"No image path found for batch item {i}")
                        self.stats["images_without_masks"] += 1
                        modified_outputs.append(output[i : i + 1])
                        continue

                    img_basename = Path(img_path).stem
                    mask_path = self._find_mask_path(img_basename)

                    if mask_path is None:
                        logging.info(f"No mask found for image: {img_basename}")
                        self.stats["images_without_masks"] += 1
                        modified_outputs.append(output[i : i + 1])
                        continue

                    logging.info(
                        f"Found mask for image: {img_basename} -> {Path(mask_path).name}"
                    )
                    self.stats["images_with_masks"] += 1

                    # Store original image path for visualization
                    if should_visualize:
                        self.visualization_data["original_input"] = img_path

                    # Load and process mask
                    mask = Image.open(mask_path).convert("L")

                    # Store original mask for visualization
                    if should_visualize:
                        self.visualization_data["mask"] = mask

                    # Resize mask to match feature map dimensions
                    feature_h, feature_w = output.shape[2], output.shape[3]
                    resized_mask = transforms.Resize(
                        (feature_h, feature_w),
                        interpolation=transforms.InterpolationMode.NEAREST,
                    )(mask)

                    # Convert to tensor and move to correct device
                    resized_mask_tensor = transforms.ToTensor()(resized_mask).to(
                        output.device
                    )

                    # Log mask statistics
                    mask_min = resized_mask_tensor.min().item()
                    mask_max = resized_mask_tensor.max().item()
                    mask_mean = resized_mask_tensor.mean().item()
                    logging.info(
                        f"Mask stats for {img_basename}: min={mask_min:.4f}, max={mask_max:.4f}, mean={mask_mean:.4f}"
                    )

                    # Store downsized mask for visualization
                    if should_visualize and layer_name:
                        self.visualization_data["downsized_masks"][
                            layer_name
                        ] = resized_mask_tensor.detach().clone()

                    # Expand mask dimensions to match output channels
                    expanded_mask = resized_mask_tensor.expand(
                        1, output.shape[1], feature_h, feature_w
                    )

                    # Apply mask to feature map
                    masked_output = output[i : i + 1] * expanded_mask
                    self.stats["masks_applied"] += 1

                    # Log feature map modification
                    if layer_name:
                        fm_before = output[i : i + 1].mean().item()
                        fm_after = masked_output.mean().item()
                        logging.info(
                            f"Feature map {layer_name} for {img_basename}: before={fm_before:.4f}, after={fm_after:.4f}"
                        )

                    # Store masked feature map for visualization
                    if should_visualize and layer_name:
                        self.visualization_data["masked_feature_maps"][
                            layer_name
                        ] = masked_output.detach().clone()

                    modified_outputs.append(masked_output)

                except Exception as e:
                    self.stats["masks_failed"] += 1
                    logging.error(f"Error applying mask: {e}")
                    # Fall back to original output on error
                    modified_outputs.append(output[i : i + 1])

            # Log summary statistics periodically
            if (
                self.stats["total_images_processed"] % 50 == 0
                and layer_name == "model.15"
            ):
                logging.info(f"MGA Stats: {self.stats}")

            # Combine modified outputs back into a batch
            if modified_outputs:
                return torch.cat(modified_outputs, dim=0)
            else:
                return output

        return hook

    # Override preprocess_batch to capture image paths
    def preprocess_batch(self, batch):
        print(f"Processing batch {self.batch_count} with keys: {batch.keys()}")

        # Store image paths before preprocessing
        if "im_file" in batch:
            self.current_batch_paths = batch["im_file"]
            # Log sample of paths
            print(f"Sample paths: {self.current_batch_paths[:3]}")

            # Update MGA trainer with these paths
            if hasattr(self, "mga_trainer"):
                self.mga_trainer.current_batch_paths = self.current_batch_paths
            else:
                print("WARNING: No mga_trainer attribute found!")

        # Increment batch count
        self.batch_count += 1

        # Print current image shape for debugging
        if "img" in batch:
            print(f"Input image shape: {batch['img'].shape}")

        # Call the original preprocessing
        result = super().preprocess_batch(batch)

        return result

    def visualize_mga_process(self, batch_idx):
        """Create a 4x4 visualization grid of the MGA process"""
        if not all(
            key in self.visualization_data for key in ["original_input", "mask"]
        ):
            print("Visualization data not available")
            return

        # Create a figure
        plt.figure(figsize=(15, 15))

        # 1. Original input image
        plt.subplot(3, 3, 1)
        input_img = Image.open(self.visualization_data["original_input"])
        plt.imshow(input_img)
        plt.title("Input Image")
        plt.axis("off")

        # 2. Original mask
        plt.subplot(3, 3, 2)
        plt.imshow(self.visualization_data["mask"], cmap="gray")
        plt.title("Segmentation Mask")
        plt.axis("off")

        # 3. Leave predictions empty or plot later if available
        plt.subplot(3, 3, 3)
        plt.title("Predictions (not implemented)")
        plt.axis("off")

        # Feature map layer names to display
        layers = ["model.15", "model.18", "model.21"]
        layer_titles = ["P3", "P4", "P5"]

        # 4-6. Feature maps before masking
        for i, (layer, title) in enumerate(zip(layers, layer_titles)):
            plt.subplot(3, 3, 4 + i)
            if layer in self.visualization_data["feature_maps"]:
                # Get the feature map and visualize the mean across channels
                feature_map = (
                    self.visualization_data["feature_maps"][layer][0].mean(0).cpu()
                )
                plt.imshow(feature_map.numpy(), cmap="viridis")
                plt.title(f"Feature Map {title}")
                plt.axis("off")

        # 7-9. Downsized masks
        for i, (layer, title) in enumerate(zip(layers, layer_titles)):

            plt.subplot(3, 3, 7 + i)
            if layer in self.visualization_data["downsized_masks"]:
                mask = self.visualization_data["downsized_masks"][layer][0].cpu()
                plt.imshow(mask.numpy(), cmap="gray")
                plt.title(f"Mask for {title}")
                plt.axis("off")

        # 10-12. Masked feature maps
        for i, (layer, title) in enumerate(zip(layers, layer_titles)):
            plt.subplot(3, 3, 10 + i)
            if layer in self.visualization_data["masked_feature_maps"]:
                # Get the masked feature map and visualize the mean across channels
                feature_map = (
                    self.visualization_data["masked_feature_maps"][layer][0]
                    .mean(0)
                    .cpu()
                )
                plt.imshow(feature_map.numpy(), cmap="viridis")
                plt.title(f"Masked FM {title}")
                plt.axis("off")

        # Save the visualization
        save_dir = os.path.join(
            "/home/mariopasc/Python/Datasets/COMBINED/detection/runs/train/mga_yolo",
            "visualizations",
        )
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"mga_visualization_batch_{batch_idx}.png"))
        plt.close()

        print(f"Visualization saved for batch {batch_idx}")

    def _find_mask_path(self, img_basename):
        """Find corresponding mask file for an image"""
        for mask_file in os.listdir(self.masks_folder):
            mask_basename = Path(mask_file).stem
            if mask_basename == img_basename:
                return os.path.join(self.masks_folder, mask_file)
        return None

    def train(self):
        """Run the training with Mask-Guided Attention"""
        # Prepare the model with MGA hooks
        model = self.register_mga_hooks(self.model)

        # Create a custom trainer that will store batch information
        # Import the base class we need to extend
        from ICA_Detection.external.ultralytics.ultralytics.models.yolo.detect.train import (
            DetectionTrainer,
        )

        class MGADetectionTrainer(DetectionTrainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.current_batch_paths = None
                self.mga_trainer = None
                self.batch_count = 0
                logging.info("MGADetectionTrainer initialized")

            # Override the _do_train method to capture batch information
            def _do_train(self, world_size=1):
                # Store reference to MGA trainer
                self.mga_trainer = mga_trainer
                logging.info(f"Starting MGA training with {world_size} GPUs")
                return super()._do_train(world_size)

            # Override preprocess_batch to capture image paths
            def preprocess_batch(self, batch):
                # Store image paths before preprocessing
                if "im_file" in batch:
                    self.current_batch_paths = batch["im_file"]
                    # Update MGA trainer with these paths
                    if hasattr(self, "mga_trainer"):
                        self.mga_trainer.current_batch_paths = self.current_batch_paths

                        # Log image paths
                        logging.debug(
                            f"Batch {self.batch_count} images: {[Path(p).name for p in self.current_batch_paths[:3]]}"
                        )

                    # Increment batch count
                    self.batch_count += 1

                    # Check if we should visualize this batch
                    should_visualize = (
                        self.batch_count % self.mga_trainer.visualize_interval == 0
                    )
                    self.mga_trainer.visualize_current_batch = should_visualize
                    self.mga_trainer.current_batch_idx = self.batch_count

                    if should_visualize:
                        logging.info(f"Will visualize batch {self.batch_count}")

                # Call the original preprocessing
                result = super().preprocess_batch(batch)

                # After batch is processed, create visualization if needed
                if (
                    hasattr(self, "mga_trainer")
                    and self.mga_trainer.visualize_current_batch
                ):
                    self.mga_trainer.visualize_mga_process(self.batch_count)
                    # Reset flag after visualization
                    self.mga_trainer.visualize_current_batch = False

                return result

        # Create a reference to self for use in the callbacks
        mga_trainer = self

        # Replace the default trainer with our custom one
        model.trainer = MGADetectionTrainer

        # Initialize training
        logging.info(
            f"Starting training with Mask-Guided Attention for {self.epochs} epochs"
        )
        logging.info(f"Masks folder: {self.masks_folder}")
        logging.info(
            f"Will generate visualizations every {self.visualize_interval} batches"
        )

        # Start training
        results = model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            imgsz=self.imgsz,
            project="/home/mariopasc/Python/Datasets/COMBINED/detection/runs/train",
            name="mga_yolo",
            device="cuda:0",
            trainer=MGADetectionTrainer,
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

        logging.info("Training complete!")
        logging.info(f"Final MGA Stats: {self.stats}")
        return model


# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "yolov8n.pt"  # Default model

    # Configuration
    data_yaml = "/home/mariopasc/Python/Datasets/COMBINED/YOLO_MGA/detection/yolo_ica_detection.yaml"
    masks_folder = "/home/mariopasc/Python/Datasets/COMBINED/YOLO_MGA/masks"

    # Initialize and run training
    mga_trainer = MaskGuidedTrainer(
        model_cfg=model_path,
        data_yaml=data_yaml,
        masks_folder=masks_folder,
        epochs=5,
        imgsz=512,
        visualize_interval=50,  # Generate visualization every 50 batches
    )

    trained_model = mga_trainer.train()
    print(f"Model saved to {trained_model.ckpt_path}")
