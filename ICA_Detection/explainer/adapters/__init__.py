# ICA_Detection/explainer/adapters/__init__.py
"""
Import each concrete adapter so its @register_adapter decorator executes.

If you add a new adapter file, just import it here and you're done.
"""

from ICA_Detection.explainer.adapters.ultralytics import UltralyticsAdapter   # registers "ultralytics"
from ICA_Detection.explainer.adapters.dca_yolov8 import DCAYOLOv8Adapter      # registers "dca_yolov8"

# Add future adapters below ↓
# from .my_new_yolo import MyNewYOLOAdapter
