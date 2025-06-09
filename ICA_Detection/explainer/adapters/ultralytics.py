"""Adapter for stock Ultralytics-YOLO ≥ v8.1"""

from typing import List

import torch
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.ops import non_max_suppression

from ICA_Detection.explainer.adapters.base import BaseAdapter
from ICA_Detection.explainer.registry import register_adapter


@register_adapter("ultralytics")
class UltralyticsAdapter(BaseAdapter):
    DEFAULT_LAYERS = [12, 17, 21]

    # -------------------------------- required impl ---------------------------

    def _load_model(self, weight, device):
        model = attempt_load_weights(weight, device)
        for p in model.parameters():
            p.requires_grad_(True)
        return model

    def _get_class_names(self):
        return self.model.names

    def get_target_layers(self) -> List[torch.nn.Module]:
        layer_ids = self.layer_ids or self.DEFAULT_LAYERS
        return [self.model.model[i] for i in layer_ids]

    def postprocess(self, preds):
        # preds is list[B, N, 6]
        return non_max_suppression(
            preds,
            conf_thres=self.conf_threshold,
            iou_thres=0.45,
        )[0]
