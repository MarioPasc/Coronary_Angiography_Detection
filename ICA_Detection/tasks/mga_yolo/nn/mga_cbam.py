import torch
import torch.nn as nn
from typing import Optional, Literal

from .cbam import CBAM


class MaskGuidedCBAM(nn.Module):
    """Mask-Guided Convolutional Block Attention Module.

    This module enhances feature maps using the CBAM architecture guided by
    attention masks. The process follows these steps:
    1. Multiply feature map F by mask M to get Fmasked = F⊗M
    2. Apply CBAM on Fmasked to get F~
    3. Fuse F~ with original features F using either element-wise
       multiplication or addition

    Reference:
        "CBAM: Convolutional Block Attention Module"
        https://arxiv.org/abs/1807.06521

    Args:
        channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for the channel attention module.
        fusion_method (str): Method to fuse masked and original features.
                             Options: "add" or "multiply". Default: "add".
    """

    def __init__(
        self,
        channels: int,
        reduction_ratio: int,
        fusion_method: Literal["add", "multiply"] = "add",
    ) -> None:
        super(MaskGuidedCBAM, self).__init__()

        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.cbam = CBAM(channels=self.channels, r=self.reduction_ratio)

        if fusion_method not in ["add", "multiply"]:
            raise ValueError(
                f"Unsupported fusion method: {fusion_method}. Choose 'add' or 'multiply'."
            )
        self.fusion_method = fusion_method

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
            return self.cbam(feature_map)

        # Ensure mask has the right shape
        if mask.dim() == 3:  # [B, H, W]
            mask = mask.unsqueeze(1)  # [B, 1, H, W]

        # Resize mask if dimensions don't match
        _, _, h, w = feature_map.shape
        _, _, mask_h, mask_w = mask.shape

        if h != mask_h or w != mask_w:
            mask = nn.functional.interpolate(mask, size=(h, w), mode="nearest")

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


class MaskGuidedCBAMHook:
    """Hook implementation for applying Mask-Guided CBAM to YOLO feature maps.

    This class can be used with PyTorch's forward hooks to apply mask-guided
    attention during model inference.
    """

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        fusion_method: Literal["add", "multiply"] = "add",
    ) -> None:
        """Initialize the hook.

        Args:
            channels (int): Number of feature map channels.
            reduction_ratio (int): Reduction ratio for CBAM. Default: 16.
            fusion_method (str): Method to fuse features. Default: "add".
        """
        self.mga_cbam = MaskGuidedCBAM(
            channels=channels,
            reduction_ratio=reduction_ratio,
            fusion_method=fusion_method,
        )

    def __call__(
        self, module: nn.Module, input_: tuple, output: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply mask-guided CBAM attention.

        Args:
            module: The PyTorch module being hooked
            input_: Input to the module
            output: Output from the module
            mask: Attention mask

        Returns:
            torch.Tensor: Enhanced feature map
        """
        return self.mga_cbam(output, mask)
