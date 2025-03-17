import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from ICA_Detection.external.ultralytics.ultralytics.engine.model import Model as YOLOModel
from ICA_Detection.external.ultralytics.ultralytics.engine.results import Results
from ICA_Detection.external.ultralytics.ultralytics.utils import ops

from .utils.npz_loader import preprocess_npz_for_yolo
from .mask_guided_yolo import MaskGuidedYOLO

class MaskedYOLOModel(YOLOModel):
    """
    Extended YOLO model that supports mask-guided attention with .npz input files.
    """
    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        # Initialize the parent YOLO model
        super().__init__(model, task, verbose)
        
        # Create the mask-guided model wrapper
        self.masked_model = MaskGuidedYOLO(model)
        
    def predict(self, source=None, stream=False, **kwargs):
        """
        Overrides the predict method to handle .npz files with masks.
        """
        # Check if source is an .npz file
        if isinstance(source, (str, Path)) and str(source).lower().endswith('.npz'):
            # Process the .npz file
            tensor = preprocess_npz_for_yolo(source)
            
            # Store original prediction function
            original_predict = self.predictor.predict
            
            # Define a wrapper for the predict function to handle our tensor
            def predict_wrapper(source=None, **kwargs):
                # If not an .npz file, use the original predict function
                if not isinstance(source, torch.Tensor) or source.shape[1] <= 3:
                    return original_predict(source, **kwargs)
                
                # Handle our tensor with image and mask
                batch = source
                
                # Run inference with our masked model
                with torch.no_grad():
                    preds = self.masked_model(batch)
                
                # Process results (this part depends on the specific YOLO version)
                # The implementation would need to convert predictions to Results objects
                
                # Placeholder for the processed results
                results = []
                for i, pred in enumerate(preds):
                    # Convert predictions to Results object
                    # This is a simplified example and would need to be adapted
                    results.append(Results(boxes=pred, orig_img=batch[i, :3].cpu().numpy()))
                
                return results
            
            # Replace the predict function temporarily
            self.predictor.predict = predict_wrapper
            
            # Call predict with our tensor
            results = super().predict(tensor, stream=stream, **kwargs)
            
            # Restore original predict function
            self.predictor.predict = original_predict
            
            return results
        
        # For non-npz files, use the original predict method
        return super().predict(source, stream=stream, **kwargs)
