from __future__ import annotations

import abc
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from ICA_Detection.explainer.utils import letterbox


class BaseAdapter(abc.ABC):
    """
    *The* place where model-specific code lives.

    Sub-classes only need to implement:
    - `_load_model`
    - `get_target_layers`
    - `postprocess`               (NMS etc.)
    """

    def __init__(
        self,
        weight: str | Path,
        device: torch.device,
        layer_ids: List[int] | None = None,
        conf_threshold: float = 0.25,
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.layer_ids = layer_ids or []
        self.model = self._load_model(weight, device)
        self.model.eval()
        self.names = self._get_class_names()
        self.colors = (
            np.random.uniform(0, 255, size=(len(self.names), 3)).astype(int)
        )

    # ------------------------------------------------------------------ required

    @abc.abstractmethod
    def _load_model(self, weight: str | Path, device: torch.device):
        pass

    @abc.abstractmethod
    def _get_class_names(self) -> List[str]:
        pass

    @abc.abstractmethod
    def get_target_layers(self) -> List[torch.nn.Module]:
        pass

    @abc.abstractmethod
    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Return tensor after NMS etc. Shape [N, 6]"""
        pass

    # ---------------------------------------------------------------- convenience

    def preprocess(
        self, image_path: str | Path
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Returns (image_np_float_0_1, tensor[B,C,H,W])
        """
        img = cv2.imread(str(image_path))
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = (
            torch.from_numpy(np.transpose(img, axes=[2, 0, 1]))
            .unsqueeze(0)
            .to(self.device)
        )
        return img, tensor
