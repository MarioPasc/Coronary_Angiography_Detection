import os
import yaml
from typing import Dict, Any
from pathlib import Path

import pandas as pd
from ultralytics import YOLO
import torch

import logging

def validate(unique_name: str, config:str, split: str = "test", device: str = "cuda:0", batch: int = 16) -> None:
    """
    Run validation using the best model on the specified data split.

    Args:
        unique_name (str): Unique name of the fold.
        split (str): Data split to validate (default is "test").
        device (str): The device to use for validation.
    """
    try:
        best_model_path = Path(f"./runs/detect/{unique_name}/weights/best.pt")
        if not best_model_path.exists():
            logging.error(f"Best model not found at {best_model_path}")
            return

        logging.info(f"Loading best model from {best_model_path}")
        model = YOLO(model=str(best_model_path), task="detect", verbose=True)

        logging.info(f"Running validation on split: {split}")
        val = model.val(
            data=config,
            imgsz=512,
            batch=batch,
            iou=0.5,
            plots=False,
            split=split,
            workers=0,
            half=True,
            device=device,
        )

        torch.cuda.empty_cache()

        results = {
            f"{split}/precision": val.box.mp,
            f"{split}/recall": val.box.mr,
            f"{split}/mAP50": val.box.map50,
            f"{split}/mAP50-95": val.box.map,
        }
        logging.info(f"Validation results for '{split}': Precision={val.box.mp}, Recall={val.box.mr}, mAP50={val.box.map50}, mAP50-95={val.box.map}")

        results_csv_path = Path(f"./results/{unique_name}_validation_metrics.csv")
        pd.DataFrame([results]).to_csv(results_csv_path, index=False)
        logging.info(f"Validation metrics saved to {results_csv_path}")
    except Exception as e:
        logging.error(f"Validation error for split {split}: {e}")
        
        
if __name__ == "__main__":
    config = ""
    unique_name = ""
    batch = 16
    split= "test"
    device = "cuda:0"
    
    
    validate(unique_name=unique_name, 
             config=config, split=split,
             device=device, batch=batch)