from typing import Dict, List, Optional, Callable, Tuple
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import os
import re

import logging


class HookManager:
    """
    Manages forward hooks for Mask-Guided Attention in neural networks.

    This class applies segmentation masks to feature maps during forward passes
    through a neural network using a skip connection approach.
    """

    def __init__(
        self,
        masks_folder: str,
        target_layers: List[str],
        get_image_path_fn: Callable[[int], Optional[str]],
        alpha: float = 0.0,
    ) -> None:
        """
        Initialize the hook manager.

        Args:
            masks_folder: Path to folder containing segmentation masks
            target_layers: List of layer names to apply MGA
            get_image_path_fn: Function to get image path from batch index
            alpha: Weight for skip connection (0.0 = only masked features,
                  1.0 = only original features)
        """
        self.masks_folder = masks_folder
        self.target_layers = target_layers
        self.get_image_path_fn = get_image_path_fn
        self.alpha = alpha
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        logging.info("HookManager initialized")

        # Batch tracking attributes
        self.current_batch_paths: List[str] = []

    def register_hooks(self, model: nn.Module) -> nn.Module:
        """
        Register MGA hooks to the model.

        Args:
            model: The YOLO model to attach hooks to

        Returns:
            The model with hooks attached
        """
        # Clear existing hooks
        self.clear_hooks()

        # Register hooks
        if isinstance(model.model, nn.Module):
            for name, module in model.model.named_modules():
                if isinstance(module, torch.nn.Module) and name.startswith("model."):
                    if name in self.target_layers:
                        # Hook to apply masks
                        mga_hook = self._get_mga_hook(name)
                        self.hooks.append(module.register_forward_hook(mga_hook))

        return model

    def clear_hooks(self) -> None:
        """Remove all registered hooks to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _apply_mask_with_skip(
        self, feature_map: torch.Tensor, mask: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """
        Apply mask to feature map with skip connection.

        Formula: output = feature_map*alpha + (1-alpha)*(feature_map * mask)

        Args:
            feature_map: Input feature map
            mask: Binary mask to apply
            alpha: Skip connection strength (0.0-1.0)

        Returns:
            Modified feature map
        """
        # Apply the mask with skip connection
        masked_feature_map = feature_map * mask
        output = feature_map * alpha + (1 - alpha) * masked_feature_map
        return output

    def _get_mga_hook(self, layer_name: Optional[str] = None) -> Callable:
        """
        Create a hook function that applies the mask to feature maps.

        Args:
            layer_name: Name of the layer for reference

        Returns:
            Hook function to be registered with PyTorch's register_forward_hook
        """

        def hook(
            module: nn.Module,
            input_feat: Tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            # The batch contains multiple images
            batch_size = output.shape[0]
            modified_outputs = []

            # Process each image in the batch
            for i in range(batch_size):
                try:
                    # Get current image path from batch
                    img_path = None
                    if hasattr(self, "current_batch_paths") and i < len(
                        self.current_batch_paths
                    ):
                        img_path = self.current_batch_paths[i]

                    if img_path is None:
                        # No image path, use original output
                        modified_outputs.append(output[i : i + 1])
                        continue

                    # Find corresponding mask
                    img_basename = Path(img_path).stem
                    mask_path = self._find_mask_path(img_basename)

                    if mask_path is None:
                        # No mask found, use original output
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

                    # Apply mask to feature map with skip connection
                    feature_map = output[i : i + 1]
                    masked_output = self._apply_mask_with_skip(
                        feature_map, expanded_mask, self.alpha
                    )

                    modified_outputs.append(masked_output)

                except Exception:
                    # Fall back to original output on error
                    modified_outputs.append(output[i : i + 1])

            # Combine modified outputs back into a batch
            if modified_outputs:
                return torch.cat(modified_outputs, dim=0)
            else:
                return output

        return hook

    def _find_mask_path(self, img_basename: str) -> Optional[str]:
        """
        Find corresponding mask file for an image.

        Args:
            img_basename: Base filename of the image without extension

        Returns:
            Full path to the mask file if found, None otherwise
        """
        try:
            mask_files = os.listdir(self.masks_folder)

            # Strategy 1: Exact match
            for mask_file in mask_files:
                mask_basename = Path(mask_file).stem
                if mask_basename == img_basename:
                    return os.path.join(self.masks_folder, mask_file)

            # Strategy 2: Partial match
            for mask_file in mask_files:
                mask_basename = Path(mask_file).stem
                if mask_basename.startswith(img_basename):
                    return os.path.join(self.masks_folder, mask_file)

            # Strategy 3: Extract numerical ID and match
            number_match = re.search(r"(\d+)$", img_basename)
            if number_match:
                number = number_match.group(1)
                for mask_file in mask_files:
                    if number in Path(mask_file).stem:
                        return os.path.join(self.masks_folder, mask_file)

            return None

        except Exception:
            return None

    def set_batch_paths(self, paths: List[str]) -> None:
        """
        Set the current batch image paths.

        Args:
            paths: List of image file paths in the current batch
        """
        self.current_batch_paths = paths.copy() if paths else []

    def set_alpha(self, alpha: float) -> None:
        """
        Set the skip connection alpha value.

        Args:
            alpha: Value between 0.0 and 1.0 (0=only masked, 1=only original)
        """
        self.alpha = max(0.0, min(1.0, alpha))  # Clamp between 0 and 1
