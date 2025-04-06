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


class MaskGuidedTrainer:
    def __init__(self, model_cfg, data_yaml, masks_folder, epochs=100, imgsz=640):
        """
        Initialize the Mask-Guided Attention trainer

        Args:
            model_cfg: Path to model config or pre-trained weights
            data_yaml: Path to data YAML file
            masks_folder: Path to folder containing masks
            epochs: Number of training epochs
            imgsz: Input image size
        """
        self.model = YOLO(model_cfg)
        self.data_yaml = data_yaml
        self.masks_folder = masks_folder
        self.epochs = epochs
        self.imgsz = imgsz

        # Load data configuration to get dataset structure
        with open(data_yaml, "r") as f:
            self.data_dict = yaml.safe_load(f)

        # Original hooks storage to prevent memory leaks
        self.hooks = []

    def register_mga_hooks(self, model):
        """Register hooks for Mask-Guided Attention to specified layers"""
        # Clear any existing hooks to prevent memory leaks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        # Register new hooks
        for name, module in model.model.named_modules():
            if isinstance(module, torch.nn.Module) and name.startswith("model."):
                if name in ["model.15", "model.18", "model.21"]:  # P3, P4, P5 features
                    print(f"Registering MGA hook for {name}")
                    self.hooks.append(module.register_forward_hook(self.get_mga_hook()))

        return model

    def get_mga_hook(self):
        """Create a hook function that applies the mask to feature maps"""

        def hook(module, input_feat, output):
            # The batch contains multiple images
            batch_size = output.shape[0]
            modified_outputs = []

            # Process each image in the batch
            for i in range(batch_size):
                # Get current image name from trainer.batch
                try:
                    # This part is tricky and depends on how ultralytics stores paths
                    # We'll need to extract them from the current batch
                    img_path = self._get_current_image_path(i)
                    if img_path is None:
                        # If we can't get the path, just use the original output
                        modified_outputs.append(output[i : i + 1])
                        continue

                    img_basename = Path(img_path).stem
                    mask_path = self._find_mask_path(img_basename)

                    if mask_path is None:
                        # If no mask is found, use the original output
                        modified_outputs.append(output[i : i + 1])
                        continue

                    # Load and process mask
                    mask = Image.open(mask_path).convert("L")

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

                    # Expand mask dimensions to match output channels
                    expanded_mask = resized_mask_tensor.expand(
                        1, output.shape[1], feature_h, feature_w
                    )

                    # Apply mask to feature map
                    masked_output = output[i : i + 1] * expanded_mask
                    modified_outputs.append(masked_output)

                except Exception as e:
                    print(f"Error applying mask: {e}")
                    # Fall back to original output on error
                    modified_outputs.append(output[i : i + 1])

            # Combine modified outputs back into a batch
            if modified_outputs:
                return torch.cat(modified_outputs, dim=0)
            else:
                return output

        return hook

    def _get_current_image_path(self, batch_idx):
        """Get current image path from the trainer's batch"""
        # This is a placeholder - the actual implementation depends on how
        # the ultralytics trainer stores image paths
        # For now, we'll need to monkey patch the trainer to store paths
        try:
            # Access the current batch info
            if hasattr(self, "current_batch_paths") and batch_idx < len(
                self.current_batch_paths
            ):
                return self.current_batch_paths[batch_idx]
            return None
        except:
            return None

    def _find_mask_path(self, img_basename):
        """Find corresponding mask file for an image"""
        for mask_file in os.listdir(self.masks_folder):
            mask_basename = Path(mask_file).stem
            if mask_basename == img_basename:
                return os.path.join(self.masks_folder, mask_file)
        return None

    def _patch_dataloader(self, trainer):
        """Patch the dataloader to capture image paths"""
        original_get_dataloader = trainer.get_dataloader

        def patched_get_dataloader(dataset_path, batch_size, *args, **kwargs):
            dataloader = original_get_dataloader(
                dataset_path, batch_size, *args, **kwargs
            )

            # Store the original batch sampler
            original_batch_sampler = dataloader.batch_sampler

            # Create a patched dataloader that records image paths
            def capture_paths_collate_fn(batch):
                # Store image paths for use in hook
                self.current_batch_paths = [item["img_path"] for item in batch]
                # Call original collate_fn
                return dataloader.collate_fn(batch)

            dataloader.collate_fn = capture_paths_collate_fn
            return dataloader

        trainer.get_dataloader = patched_get_dataloader

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
                print("Initializing MGADetectionTrainer (custom) ...")
                self.current_batch_paths = None
                self.mga_trainer = None

            # Override the _do_train method to capture batch information
            def _do_train(self, world_size=1):
                # Store reference to MGA trainer
                self.mga_trainer = mga_trainer
                return super()._do_train(world_size)

            # Override preprocess_batch to capture image paths
            def preprocess_batch(self, batch):
                # Store image paths before preprocessing
                if "im_file" in batch:
                    self.current_batch_paths = batch["im_file"]
                    # Update MGA trainer with these paths
                    if hasattr(self, "mga_trainer"):
                        self.mga_trainer.current_batch_paths = self.current_batch_paths

                # Call the original preprocessing
                return super().preprocess_batch(batch)

        # Create a reference to self for use in the callbacks
        mga_trainer = self

        # Define callback functions
        def on_train_start(trainer):
            # No need to patch dataloader now
            pass

        def on_train_batch_start(trainer):
            # No need to extract paths here, it's done in preprocess_batch
            pass

        # Add callbacks to the model
        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_batch_start", on_train_batch_start)

        # Replace the default trainer with our custom one
        model.trainer = MGADetectionTrainer

        # Initialize training
        print(f"Starting training with Mask-Guided Attention for {self.epochs} epochs")
        print(f"Masks folder: {self.masks_folder}")

        # Start training
        results = model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            imgsz=self.imgsz,
            project="/home/mariopasc/Python/Datasets/COMBINED/detection/runs/train",
            name="mga_yolo",
            trainer=MGADetectionTrainer,  # CLAVE!!!!!!!
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

        print("Training complete!")
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
        epochs=100,
        imgsz=512,
    )

    trained_model = mga_trainer.train()
    print(f"Model saved to {trained_model.ckpt_path}")
