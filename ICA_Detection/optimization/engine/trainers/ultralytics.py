"""Trainer for stock Ultralytics YOLOv8."""

from __future__ import annotations

from ICA_Detection.external.ultralytics.ultralytics import YOLO
from ICA_Detection.optimization.cfg.defaults import DEFAULT_PARAMS
from ._base import BaseTrainer


class UltralyticsTrainer(BaseTrainer):
    MODEL_CLS = YOLO
    DEFAULT_PARAMS = DEFAULT_PARAMS
