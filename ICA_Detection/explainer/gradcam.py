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

from ICA_Detection.explainer.utils import display_images, letterbox
from ICA_Detection.explainer.registry import get_adapter_cls

from cmap import Colormap

# ------------------------------------------------ helpers ---------------------


class _YOLOTarget(torch.nn.Module):
    """Grad-CAM target: sum of (class-score + bbox coords) for detections
    whose confidence ≥ self.conf.  Always returns **one scalar**.
    """

    def __init__(self, conf_thres: float, output_type: str = "all"):
        super().__init__()
        self.conf = conf_thres
        self.output_type = output_type  # "class", "box", or "all"

    # -------------------------------------------------------------------------
    def forward(self, data):
        # ---------- normalise the many YOLO return styles --------------------
        # unwrap list-of-one (Ultralytics eval mode)
        if isinstance(data, (list, tuple)) and len(data) == 1:
            data = data[0]

        # tuple(post_nms, pre_nms_boxes)  → legacy path
        if isinstance(data, (list, tuple)) and len(data) == 2:
            post_result, pre_result_boxes = data
        else:
            # plain tensor:  (N, 6)  [xyxy, conf, cls]
            post_result = data                             # (N, 6)
            pre_result_boxes = post_result[..., :4]        # (N, 4)

        if post_result.numel() == 0:                       # no detections at all
            return post_result.sum() * 0.0                 # scalar, grads OK

        # ---------- build a scalar loss -------------------------------------
        loss = torch.zeros((), device=post_result.device)  # 0-dim tensor
        for i in range(post_result.size(0)):
            if float(post_result[i, 4]) < self.conf:
                continue                                   # below threshold

            if self.output_type in ("class", "all"):
                # class confidence is column 5 in Ultralytics
                loss = loss + post_result[i, 5]

            if self.output_type in ("box", "all"):
                loss = loss + pre_result_boxes[i, :4].sum()

        # if nothing passed the threshold, keep graph alive with zero
        if loss.detach().item() == 0.0:
            loss = post_result[..., 4].sum() * 0.0

        return loss


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
        cam_threshold: float = 0.05,     # transparency gate
        pred_box_style: dict | None = None,   # fixed colour/label for PRED
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
        self.cam_threshold = cam_threshold
        # default prediction-box style (over-ride via kwarg)
        self.pred_box_style = pred_box_style or {
            "colour": (187, 85, 102),   # red, BGR for cv2
            "labelname": "Detection",
            "fontsize": 0.6,
            "linewidth": 2,
        }
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
            cam_img = show_cam_on_image(img_np, gray_cam, use_rgb=True, image_weight=0.8)

        # ---------- make low-activation pixels transparent ---------------
        if self.cam_threshold > 0:
            mask = gray_cam < self.cam_threshold        # bool H×W
            # restore the original image where CAM is weak
            cam_img[mask] = (img_np[mask] * 255).astype(np.uint8)

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
        return show_cam_on_image(img_np, renorm, use_rgb=True, colormap=Colormap("tol:nightfall"))

    def _draw_boxes(self, preds, img):
        import cv2 
        col = self.pred_box_style["colour"]
        lw  = self.pred_box_style["linewidth"]
        txt = self.pred_box_style["labelname"]
        fs  = self.pred_box_style["fontsize"]

        preds = preds.detach()
        for det in preds:
            x1, y1, x2, y2 = map(int, det[:4].cpu().numpy())
            cv2.rectangle(img, (x1, y1), (x2, y2), col, lw)
            cv2.putText(
                img, txt, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, fs, col, lw, lineType=cv2.LINE_AA
            )
        return img
