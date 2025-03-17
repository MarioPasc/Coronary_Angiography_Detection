import torch
import torch.nn as nn
from ICA_Detection.external.ultralytics.ultralytics.engine.model import Model
from ICA_Detection.external.ultralytics.ultralytics.nn.tasks import attempt_load_one_weight

from ICA_Detection.external.ultralytics.ultralytics import YOLO

from .masked_attention import MaskedAttentionLayer
from .model_adapter import InputChannelAdapter

class MaskGuidedYOLO(nn.Module):
    """
    A wrapper for YOLOv8 that incorporates mask-guided attention.
    """
    def __init__(self, yolo_model_path):
        super(MaskGuidedYOLO, self).__init__()
        
        # Load the original YOLO model
        original_model = YOLO(yolo_model_path)
        
        # Adapt the model to accept 4 channels input (RGB + mask)
        self.model = InputChannelAdapter(original_model, in_channels=4)
        
        # Store the original model's names and other metadata
        self.names = original_model.names
        self.is_adapted = True
    
    def forward(self, x):
        # x shape: [batch_size, 4, height, width] - RGB + mask
        return self.model(x)
