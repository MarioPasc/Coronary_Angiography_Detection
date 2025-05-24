"""Trainer for the DCA-YOLOv8 fork."""

from __future__ import annotations

from ICA_Detection.external.DCA_YOLOv8.DCA_YOLOv8.ultralytics.models.yolo import YOLO as DCA_YOLO
from ICA_Detection.optimization.cfg.defaults import DEFAULT_PARAMS
from ICA_Detection.optimization.engine.trainers._base import BaseTrainer


class DCAYOLOv8Trainer(BaseTrainer):
    MODEL_CLS = DCA_YOLO
    DEFAULT_PARAMS = DEFAULT_PARAMS            
