"""
Adapter for *your* fork `ICA_Detection.external.DCA_YOLOv8` that you load in
`DCAYOLOv8Trainer`.  The code is ~identical to UltralyticsAdapter – we only
import the different package so you can evolve it independently.
"""

from typing import List

import torch
from ICA_Detection.external.DCA_YOLOv8.DCA_YOLOv8.ultralytics.models.yolo import (
    YOLO as DCA_YOLO,
)
from ultralytics.utils.ops import non_max_suppression

from ICA_Detection.explainer.adapters.base import BaseAdapter
from ICA_Detection.explainer.registry import register_adapter


@register_adapter("dca_yolov8")
class DCAYOLOv8Adapter(BaseAdapter):
    DEFAULT_LAYERS = [12, 17, 21]

    # -------------------------------- required impl ---------------------------

    def _load_model(self, weight, device):
        model = DCA_YOLO(weight).model.to(device)
        for p in model.parameters():
            p.requires_grad_(True)
        return model

    def _get_class_names(self):
        return self.model.names

    def get_target_layers(self) -> List[torch.nn.Module]:
        layer_ids = self.layer_ids or self.DEFAULT_LAYERS
        return [self.model.model[i] for i in layer_ids]

    def postprocess(self, preds):
        return non_max_suppression(
            preds,
            conf_thres=self.conf_threshold,
            iou_thres=0.45,
        )[0]
