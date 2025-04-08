from __future__ import annotations
from typing import Dict, List, Optional, Callable, Tuple, Union, Protocol, TypedDict
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import os
import re
import logging
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logger = logging.getLogger("mga_yolo.hooks")

# Type definitions for better code clarity
ImagePath = str
LayerName = str


@dataclass
class MaskGuidedAttentionConfig:
    """Configuration for Mask-Guided Attention modules."""

    target_layers: List[str] = field(
        default_factory=lambda: ["model.15", "model.18", "model.21"]
    )
    reduction_ratio: int = 16
    kernel_size: int = 7  # For spatial attention convolution


class FeatureMapBundle(TypedDict):
    """Type definition for a feature map bundle."""

    original: torch.Tensor
    masked: Optional[torch.Tensor]
    layer_name: str
    image_name: Optional[str]


class HookManager:
    """
    Manages forward hooks for Mask-Guided Attention in neural networks.

    This class applies segmentation masks to feature maps during forward passes
    through a neural network using a configurable attention mechanism.
    The implementation is specifically tailored for YOLOv8's feature pyramid
    network layers (P3, P4, P5).
    """

    def __init__(
        self,
        masks_folder: str,
        target_layers: Optional[List[str]] = None,
        get_image_path_fn: Optional[Callable[[int], Optional[str]]] = None,
        config: Optional[MaskGuidedAttentionConfig] = None,
    ) -> None:
        """
        Initialize the hook manager.

        Args:
            masks_folder: Path to folder containing segmentation masks
            target_layers: List of layer names to apply MGA (default: P3, P4, P5)
            get_image_path_fn: Function to get image path from batch index
            config: Configuration for mask-guided attention
        """
        self.masks_folder = masks_folder
        self.config = config or MaskGuidedAttentionConfig()
        self.target_layers = target_layers or self.config.target_layers
        self.get_image_path_fn = get_image_path_fn
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Module cache to avoid recreating attention modules
        self._module_cache: Dict[str, nn.Module] = {}

        # Batch tracking attributes
        self.current_batch_paths: List[str] = []

        # Setup logging
        self._setup_logging()
        logger.info(
            f"[HookManager] Initialized with {len(self.target_layers)} target layers"
        )

    def _setup_logging(self) -> None:
        """Configure logging for the hook manager."""
        # Make sure root logger is configured (will work with trainer's logging)
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

        # Reset module cache
        self._module_cache = {}

        # Hook counter for logging
        hooks_registered = 0

        # Register hooks
        if hasattr(model, "model") and isinstance(model.model, nn.Module):
            for name, module in model.model.named_modules():
                if isinstance(module, torch.nn.Module) and name in self.target_layers:
                    # Hook to apply masks
                    mga_hook = self._get_mga_hook(name)
                    self.hooks.append(module.register_forward_hook(mga_hook))
                    hooks_registered += 1

        logger.info(f"[HookManager] Registered {hooks_registered} MGA hooks")
        return model

    def clear_hooks(self) -> None:
        """Remove all registered hooks to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.debug("[HookManager] Cleared all hooks")

    def __del__(self) -> None:
        """Ensure hooks are cleared when object is deleted."""
        self.clear_hooks()

    def _apply_mask_with_cbam(
        self, feature_map: torch.Tensor, mask: torch.Tensor, layer_name: str
    ) -> torch.Tensor:
        """
        Apply mask to feature map with CBAM attention.

        Implementation follows the MGA-YOLO approach:
        1. Create masked features: Fmasked = F⊗M
        2. Apply CBAM: F~ = CBAM(Fmasked)

        Args:
            feature_map: Input feature map [B,C,H,W]
            mask: Binary mask [B,C,H,W]
            layer_name: Name of the layer for module caching

        Returns:
            Modified feature map with same shape as input
        """
        print(f"DEBUG: Applying CBAM to {layer_name}")  # Direct print for debugging
        print(f"Current logger level: {logger.level}, handlers: {len(logger.handlers)}")

        logger.info(
            f"[HookManager] Applying CBAM to {layer_name} with mask shape {mask.shape}"
        )

        # Dynamic import to avoid circular imports
        from ICA_Detection.tasks.mga_yolo.nn.mga_cbam import MaskGuidedCBAM

        # Get number of channels from feature map
        channels = feature_map.shape[1]

        # Get or create CBAM module with appropriate channel count
        cache_key = f"{layer_name}_{channels}"
        if cache_key not in self._module_cache:
            self._module_cache[cache_key] = MaskGuidedCBAM(
                channels=channels,
                reduction_ratio=self.config.reduction_ratio,
            ).to(feature_map.device)

        mga_cbam = self._module_cache[cache_key]

        # Apply mask first
        masked_feature = feature_map * mask

        # Apply CBAM to masked feature
        enhanced_feature = mga_cbam(masked_feature)

        return enhanced_feature

    def _apply_mask_with_skip(
        self,
        feature_map: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply mask to feature map with simple skip connection.

        Args:
            feature_map: Input feature map
            mask: Binary mask to apply

        Returns:
            Modified feature map
        """
        return feature_map * mask

    def _get_mga_hook(self, layer_name: str) -> Callable:
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
                    elif self.get_image_path_fn:
                        img_path = self.get_image_path_fn(i)

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
                    feature_h, feature_w = output.shape[2], output.shape[3]
                    mask_tensor = self._process_mask(mask_path, (feature_h, feature_w))

                    if mask_tensor is None:
                        # Mask processing failed, use original output
                        modified_outputs.append(output[i : i + 1])
                        continue

                    # Move mask to correct device
                    mask_tensor = mask_tensor.to(output.device)

                    # Expand mask dimensions to match output channels
                    expanded_mask = mask_tensor.expand(
                        1, output.shape[1], feature_h, feature_w
                    )

                    # Apply mask to feature map with configured strategy
                    feature_map = output[i : i + 1]
                    masked_output = self._apply_mask_with_cbam(
                        feature_map, expanded_mask, layer_name
                    )

                    modified_outputs.append(masked_output)

                except Exception as e:
                    # Log error and fall back to original output
                    logger.exception(
                        f"[HookManager] Error in MGA hook for batch item {i}: {e}"
                    )
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
            if not os.path.exists(self.masks_folder):
                logger.warning(
                    f"[HookManager] Masks folder does not exist: {self.masks_folder}"
                )
                return None

            mask_files = os.listdir(self.masks_folder)

            # Strategy 1: Exact match
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                mask_path = os.path.join(self.masks_folder, f"{img_basename}{ext}")
                if os.path.exists(mask_path):
                    return mask_path

            # Strategy 2: Partial match
            for mask_file in mask_files:
                mask_basename = Path(mask_file).stem
                if mask_basename == img_basename or mask_basename.startswith(
                    img_basename
                ):
                    return os.path.join(self.masks_folder, mask_file)

            # Strategy 3: Extract numerical ID and match
            number_match = re.search(r"(\d+)$", img_basename)
            if number_match:
                number = number_match.group(1)
                for mask_file in mask_files:
                    if number in Path(mask_file).stem:
                        return os.path.join(self.masks_folder, mask_file)

            logger.debug(f"[HookManager] No mask found for image: {img_basename}")
            return None

        except Exception as e:
            logger.exception(
                f"[HookManager] Error finding mask for {img_basename}: {e}"
            )
            return None

    def _process_mask(
        self, mask_path: str, target_size: Tuple[int, int]
    ) -> Optional[torch.Tensor]:
        """
        Load and process a mask to match feature map dimensions.

        Args:
            mask_path: Path to the mask file
            target_size: Target size as (height, width)

        Returns:
            Processed mask tensor or None if processing failed
        """
        try:
            # Load mask as grayscale image
            mask = Image.open(mask_path).convert("L")

            # Resize to match feature map dimensions
            resized_mask = transforms.Resize(
                target_size, interpolation=transforms.InterpolationMode.NEAREST
            )(mask)

            # Convert to tensor [1, 1, H, W]
            mask_tensor = transforms.ToTensor()(resized_mask).unsqueeze(0)

            return mask_tensor

        except IOError:
            logger.error(f"[HookManager] Cannot open mask file: {mask_path}")
            return None
        except Exception as e:
            logger.exception(f"[HookManager] Error processing mask {mask_path}: {e}")
            return None

    def set_batch_paths(self, paths: List[str]) -> None:
        """
        Set the current batch image paths.

        Args:
            paths: List of image file paths in the current batch
        """
        self.current_batch_paths = paths.copy() if paths else []

    def set_config(self, config: MaskGuidedAttentionConfig) -> None:
        """
        Update the configuration for mask-guided attention.

        Args:
            config: New configuration
        """
        self.config = config
        # Clear module cache to ensure new settings take effect
        self._module_cache = {}

    def __repr__(self) -> str:
        """String representation of the HookManager."""
        return (
            f"HookManager(masks_folder='{self.masks_folder}', "
            f"target_layers={self.target_layers}, "
            f"active_hooks={len(self.hooks)})"
        )
