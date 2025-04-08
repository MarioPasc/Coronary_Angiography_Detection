import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).

    This module combines channel attention and spatial attention mechanisms to
    enhance feature representation by focusing on 'what' and 'where' information.

    Args:
        channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for the channel attention module.
        method (str): Method to fuse the spatial attention and channel attention outputs.
                      Options: "sequential", "concat", or "add". Default: "skipconn":
                        1. "sequential": This is the original CBAM method. It applied CAM
                           first and then SAM, over the element-wise multiplation of CAM
                           and the input features. Finally, the output of SAM is multiplied
                           with the CAM output [1].
                        2. "concat": Concatenate the outputs of CAM and SAM. Concat doubles
                           the dimension of channels [2].
                        3. "add": Add the outputs of CAM and SAM. This is a simple
                           summation of the two outputs [2].

    References:
        [1] "CBAM: Convolutional Block Attention Module"
            Access via https://arxiv.org/abs/1807.06521
        [2] "MGA-YOLOv4: a multi-scale pedestrian detection method based on mask-guided attention"
            Access via https://link.springer.com/article/10.1007/s10489-021-03061-3
    """

    def __init__(
        self,
        channels: int,
        r: int,
        method: Literal["sequential", "concat", "add"] = "sequential",
    ) -> None:
        super(CBAM, self).__init__()
        self.channels = channels
        self.reduction_ratio = r
        self.method = method
        if self.method not in ["sequential", "concat", "add"]:
            raise ValueError(
                f"Unsupported method: {self.method}. Choose 'sequential', 'concat', 'add'."
            )
        self.channel_attention = CAM(channels=self.channels, r=self.reduction_ratio)
        self.spatial_attention = SAM(bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel and spatial attention sequentially.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]

        Returns:
            torch.Tensor: Enhanced feature map with same shape as input
        """
        if self.method == "sequential":
            cam_output = self.channel_attention(x)
            cam_output = cam_output * x
            sam_output = self.spatial_attention(cam_output)
            x_refined = cam_output * sam_output
        elif self.method == "concat":
            cam_output = self.channel_attention(x)
            sam_output = self.spatial_attention(x)
            x_refined = torch.cat((cam_output, sam_output), dim=1)
        elif self.method == "add":
            cam_output = self.channel_attention(x)
            sam_output = self.spatial_attention(x)
            x_refined = cam_output + sam_output

        # Skipped connection
        if x.shape != x_refined.shape:
            raise ValueError(
                f"Input shape {x.shape} does not match refined shape {x_refined.shape}."
            )

        # We only return the attention-refined feature map
        return x_refined


class CAM(nn.Module):
    """Channel Attention Module.

    This module generates a channel attention map by exploiting both max-pooled
    and average-pooled features along the spatial dimensions.

    Args:
        channels (int): Number of input channels.
        r (int): Reduction ratio for the MLP.
    """

    def __init__(self, channels: int, r: int) -> None:
        super(CAM, self).__init__()
        if channels <= 0 or r <= 0 or channels % r != 0:
            raise ValueError(
                f"Invalid parameters: channels={channels}, r={r}. "
                f"Ensure channels > 0, r > 0, and channels is divisible by r."
            )

        # TODO: We can play with:
        #  1. Different reduction ratios (r)
        #  2. Different MLP architectures (e.g., different activation functions)

        self.channels = channels
        self.r = r
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=self.channels,
                out_features=self.channels // self.r,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.channels // self.r,
                out_features=self.channels,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]

        Returns:
            torch.Tensor: Channel-refined feature map with same shape as input
        """
        batch_size, channels, _, _ = x.size()

        # Global max pooling
        max_pool = F.adaptive_max_pool2d(x, output_size=1).view(batch_size, channels)
        # Global average pooling
        avg_pool = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size, channels)

        # Apply shared MLP to both pooled features
        max_out = self.mlp(max_pool).view(batch_size, channels, 1, 1)
        avg_out = self.mlp(avg_pool).view(batch_size, channels, 1, 1)

        # Combine and create attention map
        attention = torch.sigmoid(max_out + avg_out)

        # In the original CBAM paper, the output would be the elemt-wise multiplication
        # of the input features and the attention map
        # However, in MGA-YOLOv4, the attention map and input features are combined using
        # a skip connection. We are keeping this for now.
        # Same with SAM.

        return x + attention * x


class SAM(nn.Module):
    """Spatial Attention Module.

    This module generates a spatial attention map by utilizing max-pooled and
    average-pooled features along the channel dimension.

    Args:
        bias (bool, optional): Whether to include bias in the convolution layer. Default: False.
    """

    def __init__(self, bias: bool = False) -> None:
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=self.bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]

        Returns:
            torch.Tensor: Spatially-refined feature map with same shape as input
        """
        # Max pooling along channel dimension
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        # Average pooling along channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        # Concatenate pooled features
        concat = torch.cat((max_pool, avg_pool), dim=1)

        # Generate spatial attention map
        spatial_map = torch.sigmoid(self.conv(concat))

        return x + spatial_map * x
