import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedAttentionLayer(nn.Module):
    """
    A layer that applies attention to features based on an input mask.
    """
    def __init__(self, in_channels):
        super(MaskedAttentionLayer, self).__init__()
        self.conv_mask = nn.Conv2d(1, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask):
        # Process mask
        mask_features = self.conv_mask(mask)
        attention_weights = self.sigmoid(mask_features)
        
        # Apply attention
        return x * attention_weights

class MaskGuidedConv(nn.Module):
    """
    A convolutional layer with mask-guided attention.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(MaskGuidedConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.attention = MaskedAttentionLayer(out_channels)
        
    def forward(self, x, mask):
        # Regular convolution
        x = self.conv(x)
        
        # Apply attention using mask
        x = self.attention(x, mask)
        
        return x
