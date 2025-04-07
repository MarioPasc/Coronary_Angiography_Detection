from typing import Dict, Any, List, Optional, Callable, Tuple
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import logging
import os
import re


class HookManager:
    """
    Manages forward hooks for Mask-Guided Attention in neural networks.

    This class handles the application of segmentation masks to feature maps
    during forward passes through a neural network. It registers hooks to
    capture feature maps, apply masks, and store visualization data.
    """

    def __init__(
        self,
        masks_folder: str,
        target_layers: List[str],
        get_image_path_fn: Callable[[int], Optional[str]],
    ) -> None:
        """
        Initialize the hook manager.

        Args:
            masks_folder: Path to folder containing segmentation masks
            target_layers: List of layer names to apply MGA
            get_image_path_fn: Function to get image path from batch index
        """
        self.masks_folder = masks_folder
        self.target_layers = target_layers
        self.get_image_path_fn = get_image_path_fn
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Storage for visualization data
        self.visualization_data: Dict[str, Any] = {
            "original_input": None,
            "mask": None,
            "feature_maps": {},  # Will store feature maps before masking
            "downsized_masks": {},  # Will store resized masks for each feature map
            "masked_feature_maps": {},  # Will store feature maps after masking
            "predictions": None,  # Will store model predictions (optional)
        }

        # Statistics for tracking MGA application
        self.stats: Dict[str, int] = {
            "total_images_processed": 0,
            "images_with_masks": 0,
            "images_without_masks": 0,
            "masks_applied": 0,
            "masks_failed": 0,
        }

        # Batch tracking attributes
        self.current_batch_paths: List[str] = []
        self.visualize_current_batch: bool = False
        self.current_batch_idx: Optional[int] = None

    def register_hooks(self, model: nn.Module) -> nn.Module:
        """
        Register all necessary hooks to the model.

        This method finds target layers in the model and attaches the
        capture and MGA hooks to them.

        Args:
            model: The YOLO model to attach hooks to

        Returns:
            The model with hooks attached
        """
        # Clear existing hooks
        self.clear_hooks()

        # Find target layers
        found_layers = []
        if isinstance(model.model, nn.Module):
            for name, module in model.model.named_modules():
                if name in self.target_layers:
                    found_layers.append(name)

        logging.info(f"Found target layers: {found_layers}")

        # Register hooks
        if isinstance(model.model, nn.Module):
            for name, module in model.model.named_modules():
                if isinstance(module, torch.nn.Module) and name.startswith("model."):
                    if name in self.target_layers:
                        # Hook to capture feature maps before masking
                        capture_hook = self._get_capture_hook(name)
                        self.hooks.append(module.register_forward_hook(capture_hook))
                        logging.info(f"Registered capture hook for {name}")

                        # Hook to apply masks
                        mga_hook = self._get_mga_hook(name)
                        self.hooks.append(module.register_forward_hook(mga_hook))
                        logging.info(f"Registered MGA hook for {name}")

        return model

    def clear_hooks(self) -> None:
        """Remove all registered hooks to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logging.info(f"Cleared {len(self.hooks)} existing hooks")

    def _get_capture_hook(self, layer_name: str) -> Callable:
        """
        Create a hook function that captures feature maps before masking.

        Args:
            layer_name: Name of the layer for storing captured feature maps

        Returns:
            Hook function to be registered with PyTorch's register_forward_hook
        """

        def hook(
            module: nn.Module,
            input_feat: Tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
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
                logging.debug(
                    f"Captured feature map for {layer_name} with shape {output.shape}"
                )

            return output

        return hook

    def _get_debug_hook(self, layer_name: str) -> Callable:
        """
        Create a debug hook to verify hooks are being called.

        Args:
            layer_name: Name of the layer to debug

        Returns:
            Hook function that logs when it's called
        """

        def hook(
            module: nn.Module,
            input_feat: Tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            logging.info(
                f"DEBUG: Hook for {layer_name} was called! Output shape: {output.shape}"
            )
            return output

        return hook

    def _get_mga_hook(self, layer_name: Optional[str] = None) -> Callable:
        """
        Create a hook function that applies the mask to feature maps.

        This hook finds the corresponding mask for each image in the batch,
        resizes it to match feature map dimensions, and multiplies the
        feature map by the mask to focus attention on relevant regions.

        Args:
            layer_name: Name of the layer for storing masked feature maps

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

            # Check if we should visualize for this batch
            should_visualize_batch = (
                hasattr(self, "visualize_current_batch")
                and self.visualize_current_batch
            )

            if should_visualize_batch:
                logging.info(
                    f"MGA hook processing for visualization: layer {layer_name}, batch size {batch_size}"
                )

            # Process each image in the batch
            for i in range(batch_size):
                # Update statistics
                self.stats["total_images_processed"] += 1

                # Check if this is the first item in batch and should be visualized
                should_visualize = i == 0 and should_visualize_batch

                try:
                    # Get current image path from batch
                    img_path = None
                    if hasattr(self, "current_batch_paths") and i < len(
                        self.current_batch_paths
                    ):
                        img_path = self.current_batch_paths[i]

                        # Store for visualization if needed
                        if should_visualize:
                            self.visualization_data["original_input"] = img_path
                            logging.info(
                                f"MGA hook stored path for visualization: {img_path}"
                            )

                    if img_path is None:
                        logging.warning(f"No image path found for batch item {i}")
                        self.stats["images_without_masks"] += 1
                        modified_outputs.append(output[i : i + 1])
                        continue

                    # Find corresponding mask
                    img_basename = Path(img_path).stem
                    mask_path = self._find_mask_path(img_basename)

                    if mask_path is None:
                        logging.warning(f"No mask found for {img_basename}")
                        self.stats["images_without_masks"] += 1
                        modified_outputs.append(output[i : i + 1])
                        continue

                    self.stats["images_with_masks"] += 1

                    # Load and process mask
                    mask = Image.open(mask_path).convert("L")

                    # Store original mask for visualization
                    if should_visualize:
                        self.visualization_data["mask"] = mask
                        logging.info(f"Stored mask for visualization: {mask_path}")

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
                    logging.debug(
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
                        logging.debug(
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
                and layer_name == self.target_layers[0]  # First target layer
            ):
                logging.info(f"MGA Stats: {self.stats}")

            # Combine modified outputs back into a batch
            if modified_outputs:
                return torch.cat(modified_outputs, dim=0)
            else:
                return output

        return hook

    def _find_mask_path(self, img_basename: str) -> Optional[str]:
        """
        Find corresponding mask file for an image.

        This method attempts several strategies to find a mask:
        1. Direct match of basename
        2. Partial match (e.g., image name is prefix of mask name)
        3. Numerical ID match (extract numbers and match those)

        Args:
            img_basename: Base filename of the image without extension

        Returns:
            Full path to the mask file if found, None otherwise
        """
        logging.debug(f"Looking for mask with basename: {img_basename}")

        try:
            mask_files = os.listdir(self.masks_folder)

            # Strategy 1: Exact match
            for mask_file in mask_files:
                mask_basename = Path(mask_file).stem
                if mask_basename == img_basename:
                    mask_path = os.path.join(self.masks_folder, mask_file)
                    logging.debug(f"Found exact match mask: {mask_path}")
                    return mask_path

            # Strategy 2: Partial match (image name is prefix of mask name)
            for mask_file in mask_files:
                mask_basename = Path(mask_file).stem
                if mask_basename.startswith(img_basename):
                    mask_path = os.path.join(self.masks_folder, mask_file)
                    logging.debug(f"Found partial match mask: {mask_path}")
                    return mask_path

            # Strategy 3: Extract numerical ID and match
            number_match = re.search(r"(\d+)$", img_basename)
            if number_match:
                number = number_match.group(1)
                for mask_file in mask_files:
                    if number in Path(mask_file).stem:
                        mask_path = os.path.join(self.masks_folder, mask_file)
                        logging.debug(f"Found number match mask: {mask_path}")
                        return mask_path

            # No match found
            logging.warning(f"No matching mask found for {img_basename}")
            if mask_files:
                logging.debug(
                    f"Available masks (first 5): {mask_files[:min(5, len(mask_files))]}"
                )
            return None

        except Exception as e:
            logging.error(f"Error searching for mask: {e}")
            return None

    def set_batch_paths(self, paths: List[str]) -> None:
        """
        Set the current batch image paths.

        Args:
            paths: List of image file paths in the current batch
        """
        self.current_batch_paths = paths.copy() if paths else []
        logging.debug(
            f"Set current_batch_paths with {len(self.current_batch_paths)} items"
        )

    def set_visualization_state(self, should_visualize: bool, batch_idx: int) -> None:
        """
        Set the state for visualization of the current batch.

        Args:
            should_visualize: Whether to visualize this batch
            batch_idx: Current batch index
        """
        self.visualize_current_batch = should_visualize
        self.current_batch_idx = batch_idx

        if should_visualize:
            logging.info(f"Will visualize batch {batch_idx}")
            # Pre-store the first image path for visualization if available
            if self.current_batch_paths:
                self.visualization_data["original_input"] = self.current_batch_paths[0]

                # Try to find and load the mask
                img_basename = Path(self.current_batch_paths[0]).stem
                mask_path = self._find_mask_path(img_basename)
                if mask_path:
                    try:
                        self.visualization_data["mask"] = Image.open(mask_path).convert(
                            "L"
                        )
                        logging.info(f"Pre-loaded mask: {mask_path}")
                    except Exception as e:
                        logging.error(f"Failed to pre-load mask: {e}")

    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get the visualization data collected by the hooks.

        Returns:
            Dictionary containing all visualization data
        """
        return self.visualization_data

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about mask application.

        Returns:
            Dictionary of mask application statistics
        """
        return self.stats
