# CADICA_Detection/__init__.py

import os
import sys

# Add external/ultralytics to the sys.path to ensure our custom ultralytics is loaded
external_path = os.path.join(os.path.dirname(__file__), "external", "ultralytics")
if external_path not in sys.path:
    sys.path.insert(0, external_path)

# Now we can import from our modified ultralytics
from .external.ultralytics.ultralytics import YOLO
