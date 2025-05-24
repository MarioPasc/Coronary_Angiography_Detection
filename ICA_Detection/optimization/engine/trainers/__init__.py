from ICA_Detection.optimization.engine.trainers.ultralytics import UltralyticsTrainer
from ICA_Detection.optimization.engine.trainers.dca_yolov8 import DCAYOLOv8Trainer 

__all__ = ["get_trainer"]

def get_trainer(source: str):
    if source == "ultralytics":
        return UltralyticsTrainer
    if source == "dca":
        return DCAYOLOv8Trainer
    raise ValueError(f"Unknown model_source {source!r}")
