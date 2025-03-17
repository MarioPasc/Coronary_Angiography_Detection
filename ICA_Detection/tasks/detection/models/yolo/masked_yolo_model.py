import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from ICA_Detection.external.ultralytics.ultralytics.engine.model import Model as YOLOModel
from ICA_Detection.external.ultralytics.ultralytics.engine.results import Results
from ICA_Detection.external.ultralytics.ultralytics.utils import ops
from ICA_Detection.external.ultralytics.ultralytics import YOLO

from .utils.npz_loader import preprocess_npz_for_yolo
from .mask_guided_yolo import MaskGuidedYOLO

class MaskedYOLOModel:
    """
    Extended YOLO model that supports mask-guided attention with .npz input files.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.masked_model = MaskGuidedYOLO(model_path)
        self.original_model = YOLO(model_path)  # Keep original for non-mask cases
        
    def predict(self, source=None, **kwargs):
        """
        Wrapper for predict that handles both standard input and image+mask pairs
        
        For .npz files, it expects:
        - 'image': RGB image array
        - 'mask': Single-channel mask array
        """
        # Store the original predict method from the YOLO model
        original_predict = self.original_model.predict
        
        # If source is an .npz file or a list/tuple of images and masks
        if isinstance(source, str) and source.endswith('.npz'):
            # Load the .npz file
            data = np.load(source)
            image = data['image']
            mask = data['mask']
            
            # Process and normalize image and mask
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0
            
            # Ensure mask is single channel and normalized
            if mask.ndim > 2:
                mask = mask[..., 0]  # Take first channel if multi-channel
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=2)  # Add channel dimension
            if mask.dtype != np.float32:
                mask = mask.astype(np.float32)
            
            # Concatenate image and mask
            x = np.concatenate([image, mask], axis=2)  # [H, W, 4]
            
            # Convert to torch tensor and adjust dimensions
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # [1, 4, H, W]
            
            # Run inference with our masked model
            with torch.no_grad():
                preds = self.masked_model(x)
            
            # Process results
            results = []
            processed_pred = ops.non_max_suppression(
                preds[0],  # First element if batched
                conf_thres=kwargs.get('conf', 0.25),
                iou_thres=kwargs.get('iou', 0.45),
                classes=kwargs.get('classes', None),
                max_det=kwargs.get('max_det', 300)
            )[0]  # Get first element since we're processing one image at a time
            
            # Create Results object
            orig_img = image if image.max() > 1.0 else (image * 255).astype(np.uint8)
            result = Results(
                boxes=processed_pred,
                orig_img=orig_img,
                names=self.masked_model.names
            )
            results = [result]
            
            return results
        
        # Otherwise, use the original predict
        return original_predict(source, **kwargs)
