import torch
import torch.nn as nn
from typing import Optional, Literal

from .cbam import CBAM

import logging

logger = logging.getLogger("mga_training.log")


class MaskGuidedCBAM(nn.Module):
    """Mask-Guided Convolutional Block Attention Module.

    This module enhances feature maps using the CBAM architecture guided by
    attention masks. The process follows these steps:
    1. Multiply feature map F by mask M to get Fmasked = F⊗M
    2. Apply CBAM on Fmasked to get F~
    3. Fuse F~ with original features F using either element-wise
       multiplication or addition

    Args:
        channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for the channel attention module.
        sam_cam_fusion (str): Method to fuse the spatial attention and channel attention outputs.
                             Options: "sequential", "concat", or "add". Default: "sequential".
                             For more details, refer to the CBAM class.
        mga_pyramid_fusion (str): Method to fuse masked and original features.
                          Let's say we have Pn {P3,P4,P5} as the feature map,
                          M as the mask. We perform Fmasked = Pn⊗M.
                          Then we apply CBAM on Fmasked to get F~.
                          Finally, we fuse F~ with original features Pn using
                          either element-wise multiplication or addition.
                          Options: "add" or "multiply". Default: "multiply".
                            - If "add", the output will be F~ + Pn.
                            - If "multiply", the output will be F~ ⊗ Pn.

    """

    def __init__(
        self,
        channels: int,
        reduction_ratio: int,
        sam_cam_fusion: Literal["sequential", "concat", "add"],
        mga_pyramid_fusion: Literal["add", "multiply"],
    ) -> None:
        super(MaskGuidedCBAM, self).__init__()

        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.cbam = CBAM(
            channels=self.channels,
            r=self.reduction_ratio,
            sam_cam_fusion=sam_cam_fusion,
        )

        if mga_pyramid_fusion not in ["add", "multiply"]:
            raise ValueError(
                f"Unsupported fusion method: {mga_pyramid_fusion}. Choose 'add' or 'multiply'."
            )
        self.fusion_method = mga_pyramid_fusion
        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging for the hook manager."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def forward(
        self, feature_map: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply mask-guided attention to feature maps.

        Args:
            feature_map (torch.Tensor): Input tensor of shape [B, C, H, W]
            mask (torch.Tensor, optional): Binary mask of shape [B, 1, H, W].
                                          If None, CBAM is applied directly to the feature_map.

        Returns:
            torch.Tensor: Enhanced feature map with same shape as input
        """
        # If no mask is provided, just apply CBAM to the original feature map
        if mask is None:
            print("Warning: No mask provided. Applying CBAM directly to feature map.")
            return self.cbam(feature_map)

        # Ensure mask is on the same device as feature map
        mask = mask.to(feature_map.device)  # type: ignore

        # Step 1: Multiply feature map by mask
        masked_features = feature_map * mask  # Fmasked = F⊗M

        # Step 2: Apply CBAM on masked features
        enhanced_features = self.cbam(masked_features)  # F~

        # Step 3: Fuse enhanced features with original features
        if self.fusion_method == "add":
            output = enhanced_features + feature_map  # F~ + F
        else:  # multiply
            output = enhanced_features * feature_map  # F~ ⊗ F

        return output
