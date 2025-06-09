#!/usr/bin/env python
"""
Generate a Grad-CAM montage for every cross-validation fold.

For each fold we build a grid  (rows = model sizes, cols = optimisers) and
save one PNG per fold:

    out_dir/
        fold_0_cam.png
        fold_1_cam.png
        fold_2_cam.png
        …

Usage
-----
python scripts/explainer/optimization_gradcam.py \
    --root /media/mpascual/PortableSSD/Coronariografías/CompBioMed/bho_compbiomed/cadica/kfold/kfold_results \
    --images /home/mpascual/research/datasets/angio/tasks/stenosis_detection/images \
    --labels /home/mpascual/research/datasets/angio/tasks/stenosis_detection/labels/yolo \
    --name cadica_p22_v14_00021 \
    --adapter ultralytics \
    --add-cbar \
    --out /media/mpascual/PortableSSD/Coronariografías/CompBioMed/bho_compbiomed/cadica/figures/gradcam \
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch

# 3rd-party (your new explainer package)
from ICA_Detection.explainer import build_explainer, utils as xutils

# --------------------------------------------------------------------------- CLI


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", required=True, type=Path, help="results root")
    p.add_argument("--images", required=True, type=Path, help="image folder")
    p.add_argument("--labels", required=True, type=Path, help="label folder (YOLO txt)")
    p.add_argument("--name", required=True, help="image / label stem (without suffix)")
    p.add_argument("--adapter", default="ultralytics", choices=["ultralytics", "dca_yolov8"])
    p.add_argument("--out", required=True, type=Path, help="output directory")
    p.add_argument("--alpha", type=float, default=0.5, help="CAM overlay opacity")
    p.add_argument("--conf-thres", type=float, default=0.25, help="confidence for predictions")
    p.add_argument("--add-cbar", action="store_true", help="add horizontal colour-bar")
    p.add_argument("--cam-method", default="eigencam")
    return p.parse_args()


# ---------------------------------------------------------------- utilities


def collect_layout(root: Path) -> tuple[list[str], list[str], int]:
    """
    Returns (model_sizes, optimisers, n_folds)

        root/
           yolov8l/{opt}/out/fold_k/weights/best.pt
           yolov8m/...
           yolov8s/...
    """
    model_sizes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not model_sizes:
        raise FileNotFoundError(f"No model folders inside {root}")

    # assume the first model folder is representative
    first_model = root / model_sizes[0]
    optimisers = sorted(d.name for d in first_model.iterdir() if d.is_dir())
    if not optimisers:
        raise FileNotFoundError(f"No optimiser folders inside {first_model}")

    # assume optimiser/out/fold_* exists and count folds
    sample_opt_out = next((first_model / optimisers[0]).glob("*/fold_*"), None)
    if sample_opt_out is None:
        raise FileNotFoundError("Could not detect any fold directories.")
    n_folds = len(list((first_model / optimisers[0] / "out").glob("fold_*")))
    return model_sizes, optimisers, n_folds


def read_ground_truth(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """
    Converts YOLO-txt labels to pixel xyxy. Returns array[N,4]
    """
    if not label_path.exists():
        return np.empty((0, 4), dtype=int)
    boxes = []
    with label_path.open() as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, xc, yc, w, h = map(float, parts)
            # denormalise
            xc, yc, w, h = xc * img_w, yc * img_h, w * img_w, h * img_h
            x1 = int(xc - w / 2)
            y1 = int(yc - h / 2)
            x2 = int(xc + w / 2)
            y2 = int(yc + h / 2)
            boxes.append((x1, y1, x2, y2))
    return np.asarray(boxes, dtype=int)


def draw_boxes(img: np.ndarray, boxes: np.ndarray, colour: tuple, thickness: int = 2):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)


def plot_fold(
    fold_idx: int,
    root: Path,
    model_sizes: list[str],
    optimisers: list[str],
    explainer_kwargs: dict,
    img_path: Path,
    gt_boxes: np.ndarray,
    alpha: float,
    add_cbar: bool,
    out_dir: Path,
):
    n_rows, n_cols = len(model_sizes), len(optimisers)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    for r, model_size in enumerate(model_sizes):
        for c, opt in enumerate(optimisers):
            weight = (
                root
                / model_size
                / opt
                / "out"
                / f"fold_{fold_idx}"
                / "weights"
                / "best.pt"
            )
            if not weight.exists():
                axes[r, c].set_title("missing")
                axes[r, c].axis("off")
                continue

            cam = build_explainer(weight=weight, **explainer_kwargs)

            cam_imgs = cam(img_path)
            cam_img = np.array(cam_imgs[0])

            # draw GT boxes in magenta (prediction boxes already drawn inside)
            draw_boxes(cam_img, gt_boxes, colour=(255, 0, 255), thickness=2)

            axes[r, c].imshow(cam_img)
            axes[r, c].set_axis_off()
            if r == 0:
                axes[r, c].set_title(opt, fontsize=10)
            if c == 0:
                axes[r, c].text(
                    -5,
                    0.5,
                    model_size,
                    va="center",
                    ha="right",
                    rotation=90,
                    fontsize=10,
                    transform=axes[r, c].transAxes,
                )

    if add_cbar:
        # create fake mappable so the colour-bar matches jet colormap 0-1
        from matplotlib.cm import ScalarMappable, get_cmap

        sm = ScalarMappable(cmap=get_cmap("jet"))
        sm.set_array([0.0, 1.0])
        fig.colorbar(sm, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.04)

    out_file = out_dir / f"fold_{fold_idx}_cam.png"
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    logging.info("Saved %s", out_file)


# --------------------------------------------------------------------------- main


def main():
    args = get_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    img_path = args.images / f"{args.name}.png"
    label_path = args.labels / f"{args.name}.txt"
    if not img_path.exists():
        sys.exit(f"Image {img_path} not found.")
    img_np = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    gt_boxes = read_ground_truth(label_path, img_np.shape[1], img_np.shape[0])

    model_sizes, optimisers, n_folds = collect_layout(args.root)
    logging.info(
        "Detected %d model sizes %s | %d optimisers %s | %d folds",
        len(model_sizes),
        model_sizes,
        len(optimisers),
        optimisers,
        n_folds,
    )

    args.out.mkdir(parents=True, exist_ok=True)

    explainer_kwargs = dict(
        model_name=args.adapter,
        method=args.cam_method,
        conf_threshold=args.conf_thres,
        show_box=True,
    )

    for fold in range(n_folds):
        plot_fold(
            fold,
            args.root,
            model_sizes,
            optimisers,
            explainer_kwargs,
            img_path,
            gt_boxes,
            args.alpha,
            args.add_cbar,
            args.out,
        )


if __name__ == "__main__":
    torch.set_grad_enabled(True)  # grad-cam needs gradients
    main()
