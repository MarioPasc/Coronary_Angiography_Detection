"""
One single, model-agnostic Grad-CAM runner.

It relies on an *Adapter* for every model family, so the code below never has
to be touched again.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import (
    EigenCAM,
    EigenGradCAM,
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    LayerCAM,
    RandomCAM,
    XGradCAM,
)
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image

from ICA_Detection.explainer.utils import display_images
from ICA_Detection.explainer.registry import get_adapter_cls


# ------------------------------------------------ helpers ---------------------


class _YOLOTarget(torch.nn.Module):
    """Classic YOLO target for grad-cam – sum over objectness + bbox coords."""

    def __init__(self, conf_thres: float, output_type: str = "all"):
        """
        output_type ∈ {"class", "box", "all"}
        """
        super().__init__()
        self.conf = conf_thres
        self.output_type = output_type

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in range(post_result.size(0)):
            if float(post_result[i].max()) >= self.conf:
                if self.output_type in ("class", "all"):
                    result.append(post_result[i].max())
                if self.output_type in ("box", "all"):
                    for j in range(4):
                        result.append(pre_post_boxes[i, j])
        return sum(result)


# ------------------------------------------------ main class ------------------


class GradCAMExplainer:
    """
    Generic Grad-CAM runner.

    Parameters
    ----------
    adapter_cls
        Any subclass of `BaseAdapter`.
    weight
        Path to `.pt` / `.pth`.
    method
        One of `"EigenGradCAM"`, `"GradCAM++"`, … (default `"EigenGradCAM"`).
    layer_ids
        Optional override for the layers we probe.
    device
        torch device.  The default chooses GPU when available.
    conf_threshold
        Only detections above this are kept for CAM & drawing.
    ratio
        Top-k ratio to keep inside `_YOLOTarget`.
    show_box
        Draw YOLO boxes on top of heat-map.
    renormalize
        If *True* the CAM is renormalised per bounding-box instead of the
        whole image.
    """

    _CAM_METHODS = {
        "gradcam": GradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "hirescam": HiResCAM,
        "xgradcam": XGradCAM,
        "gradcamplusplus": GradCAMPlusPlus,
        "layercam": LayerCAM,
        "randomcam": RandomCAM,
    }

    def __init__(
        self,
        adapter_cls,
        weight: str | Path,
        method: str = "EigenGradCAM",
        *,
        layer_ids: Sequence[int] | None = None,
        device: torch.device | None = None,
        conf_threshold: float = 0.25,
        ratio: float = 0.02,
        show_box: bool = True,
        renormalize: bool = False,
    ):
        #  ---------- bind + sanity -------------------------------------------------
        device = (
            device
            if device is not None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        method_key = method.lower()
        if method_key not in self._CAM_METHODS:
            raise ValueError(
                f"CAM method '{method}' unknown. "
                f"Choose one of {list(self._CAM_METHODS.keys())}"
            )

        #  ---------- model via adapter --------------------------------------------
        self.adapter = adapter_cls(
            weight=weight,
            device=device,
            layer_ids=list(layer_ids) if layer_ids else None,
            conf_threshold=conf_threshold,
        )

        #  ---------- grad-cam plumbing --------------------------------------------
        target_layers = self.adapter.get_target_layers()
        cam_cls = self._CAM_METHODS[method_key]
        self.cam = cam_cls(
            self.adapter.model,
            target_layers,
            use_cuda=device.type == "cuda",
        )
        # Over-ride activations so we can keep YOLO-specific post-processing
        self.cam.activations_and_grads = ActivationsAndGradients(
            self.adapter.model, target_layers, reshape_transform=None
        )

        self._target = _YOLOTarget(conf_threshold, "all")
        self.ratio = ratio
        self.show_box = show_box
        self.renormalize = renormalize

    # ------------------------------------------------------------------------- API

    def __call__(self, path: str | Path):
        """Same signature as old `yolov8_heatmap` – returns **list of PIL Images**."""
        path = Path(path)
        if path.is_dir():
            return [self._process(p) for p in sorted(path.iterdir())]
        else:
            return [self._process(path)]

    # ------------------------------------------------------------------- internals

    def _process(self, img_path: Path):
        img_np, tensor = self.adapter.preprocess(img_path)

        # grad-cam
        gray_cam = self.cam(tensor, targets=[self._target])[0, :]

        # forward pass for drawing + renorm
        preds = self.adapter.postprocess(self.adapter.model(tensor)[0])

        # heat-map → rgb uint8
        if self.renormalize:
            cam_img = self._renormalize_cam(
                preds[:, :4].cpu().numpy().astype(np.int32),
                img_np,
                gray_cam,
            )
        else:
            cam_img = show_cam_on_image(img_np, gray_cam, use_rgb=True)

        if self.show_box and len(preds) > 0:
            cam_img = self._draw_boxes(preds, cam_img)

        return Image.fromarray(cam_img)

    # ----------------------------------------------------------------- utilities

    def _renormalize_cam(self, boxes, img_np, gray_cam):
        renorm = np.zeros_like(gray_cam, dtype=np.float32)
        from pytorch_grad_cam.utils.image import scale_cam_image

        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2 = min(gray_cam.shape[1] - 1, x2)
            y2 = min(gray_cam.shape[0] - 1, y2)
            renorm[y1:y2, x1:x2] = scale_cam_image(gray_cam[y1:y2, x1:x2].copy())

        renorm = scale_cam_image(renorm)
        return show_cam_on_image(img_np, renorm, use_rgb=True)

    def _draw_boxes(self, preds, img):
        import cv2

        for det in preds.detach():
            det = det.cpu().numpy()
            cls = int(det[5])
            x1, y1, x2, y2 = map(int, det[:4])
            cv2.rectangle(
                img, (x1, y1), (x2, y2), tuple(int(c) for c in self.adapter.colors[cls]), 2
            )
            cv2.putText(
                img,
                self.adapter.names[cls],
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                tuple(int(c) for c in self.adapter.colors[cls]),
                2,
                lineType=cv2.LINE_AA,
            )
        return img
