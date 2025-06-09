"""Public API for ICA_Detection.explainer."""

from ICA_Detection.explainer.registry import build_explainer                       # factory
from ICA_Detection.explainer.utils import letterbox, display_images                # re-export helpers
from ICA_Detection.explainer.adapters import (  # ensure adapters are registered
    UltralyticsAdapter,  # registers "ultralytics"
    DCAYOLOv8Adapter,     # registers "dca_yolov8"
    # Add future adapters here ↓
    # MyNewYOLOAdapter,  # registers "my_new_yolo"
)

__all__ = [
    "build_explainer",
    "letterbox",
    "display_images",
]


import logging

# Global logger for the entire optimization module
LOGGER = logging.getLogger("ica.explainer")
LOGGER.setLevel(logging.INFO)

# Configure a default stream handler (can be overridden in cli or main)
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
_handler.setFormatter(_formatter)
LOGGER.addHandler(_handler)
