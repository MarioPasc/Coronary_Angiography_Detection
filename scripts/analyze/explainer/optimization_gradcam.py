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
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.gridspec import GridSpec

from ICA_Detection.explainer import build_explainer, LOGGER, letterbox

from cmap import Colormap
# --------------------------------------------------------------------------- CLI


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", required=True, type=Path, help="results root")
    p.add_argument("--images", required=True, type=Path, help="image folder")
    p.add_argument("--labels", required=True, type=Path, help="label folder (YOLO txt)")
    p.add_argument("--name", required=True, help="image / label stem (without suffix)")
    p.add_argument(
        "--adapter", default="ultralytics", choices=["ultralytics", "dca_yolov8"]
    )
    p.add_argument("--out", required=True, type=Path, help="output directory")
    p.add_argument("--conf-thres", type=float, default=0.1, help="confidence threshold")
    p.add_argument("--add-cbar", action="store_true", help="add colour-bar")
    p.add_argument("--add-opt-str", action="store_true", help="add optimiser string to title")
    p.add_argument("--cam-method", default="eigencam")
    p.add_argument("--add-legend", action="store_true",
                   help="draw legend for GT / PRED boxes")
    return p.parse_args()

# ---------------------------------------------------------------- bbox styles
BBOX_CONF = {
    "GT":   {"colour": (0,68,136), "labelname": "",   "fontsize": 0.9, "linewidth": 4},
    "PRED": {"colour": (204,51,17), "labelname": "", "fontsize": 0.9, "linewidth": 4},
}


# ---------------------------------------------------------------- utilities


def collect_layout(root: Path):
    sizes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not sizes:
        raise FileNotFoundError(f"No model folders inside {root}")

    optim = sorted((root / sizes[0]).iterdir())
    optim = [d.name for d in optim if d.is_dir()]
    if not optim:
        raise FileNotFoundError(f"No optimiser folders inside {root/sizes[0]}")

    n_folds = len(list((root / sizes[0] / optim[0] / "out").glob("fold_*")))
    return sizes, optim, n_folds


def read_yolo_txt(txt: Path, img_np: np.ndarray, new_shape=(640, 640)) -> np.ndarray:
    """
    Return GT boxes **in the 640×640 letter-boxed coordinate system**
    so they align with the overlay created inside GradCAMExplainer.
    """
    if not txt.exists():
        return np.empty((0, 4), dtype=int)

    h0, w0 = img_np.shape[:2]                      # original size
    _, ratio, (dw, dh) = letterbox(img_np, new_shape, auto=False)

    out = []
    for line in txt.read_text().splitlines():
        cls, xc, yc, bw, bh = map(float, line.split())
        # 1. denormalise to original pixels
        xc, yc, bw, bh = xc * w0, yc * h0, bw * w0, bh * h0
        x1, y1 = xc - bw / 2, yc - bh / 2
        x2, y2 = xc + bw / 2, yc + bh / 2
        # 2. map through letter-box
        rw, rh = ratio            # unpack tuple
        x1 = int(x1 * rw + dw);  y1 = int(y1 * rh + dh)
        x2 = int(x2 * rw + dw);  y2 = int(y2 * rh + dh)
        out.append((x1, y1, x2, y2))

    return np.asarray(out, dtype=int)


def draw_boxes(img: np.ndarray, boxes: np.ndarray, colour=(255, 0, 255), thick=2):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick)


# ---------------------------------------------------------------- plotting


def plot_fold(
    fold: int,
    root: Path,
    sizes: list[str],
    optimisers: list[str],
    explainer_kw: dict,
    img_path: Path,
    gt_boxes: np.ndarray,
    add_cbar: bool,
    add_legend: bool,
    add_opt_str: bool,
    out_dir: Path,
):
    n_rows, n_cols = len(sizes), len(optimisers)

    # -- GridSpec: gives us equal-size cells & space for c-bar --------------
    height = 3.4 * n_rows + (0.7 if add_cbar else 0)
    width = 3.4 * n_cols
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    grid = GridSpec(n_rows, n_cols, figure=fig)

    # prepare axes array for colour-bar later
    axes = np.empty((n_rows, n_cols), dtype=object)

    for r, size in enumerate(sizes):
        for c, opt in enumerate(optimisers):
            ax = fig.add_subplot(grid[r, c])
            axes[r, c] = ax

            w_path = (
                root / size / opt / "out" / f"fold_{fold}" / "weights" / "best.pt"
            )
            if not w_path.exists():
                ax.set_axis_off()
                ax.set_title("missing", fontsize=17)
                continue

            cam = build_explainer(weight=w_path, **explainer_kw)

            overlay = np.array(cam(img_path)[0])
            draw_boxes(
                overlay,
                gt_boxes,
                colour=BBOX_CONF["GT"]["colour"],
                thick=BBOX_CONF["GT"]["linewidth"],
            )

            ax.imshow(overlay)
            ax.set_axis_off()

            # columns – upper row only
            if r == 0 and add_opt_str:
                ax.set_title(opt.upper(), fontsize=20, pad=6)

            # rows – first column only
            if c == 0:
                row_lbl = size.replace("yolo", "")  # v8l|v8m|v8s
                if "dca" in size:
                    row_lbl = size.replace("dca_yolo", "")  # v8l|v8m|v8s
                ax.annotate(
                    row_lbl,
                    xy=(-0.05, 0.5),
                    xycoords="axes fraction",
                    va="center",
                    ha="right",
                    rotation=90,
                    fontsize=20,
                )

    # -------------- colour-bar under the whole grid -------------------------
    if add_cbar:
        sm = ScalarMappable(cmap=Colormap('tol:nightfall').to_mpl())
        sm.set_array([0.0, 1.0])
        cax = fig.add_axes([0.1, -0.05, 0.8, 0.03])  # [left, bottom, width, height]
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=20)

    # ---------------------------------------------------------------- legend
    if add_legend:
        import matplotlib.patches as mpatches

        def _patch(style):
            rgb = tuple(c / 255 for c in style["colour"])  # BGR→RGB + 0-1
            return mpatches.Patch(
                facecolor='none',
                edgecolor=rgb,
                linewidth=style["linewidth"],
                label=style["labelname"],
            )

        BBOX_CONF["GT"]["labelname"] = "Ground Truth"
        BBOX_CONF["PRED"]["labelname"] = "Predicted"
        handles = [_patch(BBOX_CONF["GT"]), _patch(BBOX_CONF["PRED"])]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=2,
            frameon=False,
            fontsize=20,
        )
        BBOX_CONF["GT"]["labelname"] = ""
        BBOX_CONF["PRED"]["labelname"] = ""

    out_dir = out_dir / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = "pdf"
    fname = out_dir / f"fold_{fold}_cam_{img_path.stem}.{fmt}"

    fig.savefig(fname, dpi=150, bbox_inches="tight", format=fmt)
    plt.close(fig)
    LOGGER.info("Saved %s", fname)


# --------------------------------------------------------------------------- main


def main():
    args = get_args()

    img_path = args.images / f"{args.name}.png"
    lbl_path = args.labels / f"{args.name}.txt"
    if not img_path.exists():
        sys.exit(f"Image {img_path} not found.")
    img_np = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    gt_boxes = read_yolo_txt(lbl_path, img_np)      # ← passes full image

    sizes, optim, n_folds = collect_layout(args.root)
    LOGGER.info("Detected %d model sizes %s | %d optimisers %s | %d folds", len(sizes), sizes, len(optim), optim, n_folds)

    explainer_kw = dict(
        model_name=args.adapter,
        method=args.cam_method,
        conf_threshold=args.conf_thres,
        show_box=True,
        cam_threshold=0.2,  
        pred_box_style=BBOX_CONF["PRED"], # unified colour/label
    )

    for k in range(n_folds):
        plot_fold(
            fold = k,
            root = args.root,
            sizes = sizes,
            optimisers = optim,
            explainer_kw = explainer_kw,
            img_path = img_path,
            gt_boxes = gt_boxes,
            add_cbar = args.add_cbar,
            add_legend = args.add_legend,
            out_dir = args.out,
            add_opt_str = args.add_opt_str,
        )


if __name__ == "__main__":
    torch.set_grad_enabled(True)  # grad-cam needs gradients
    main()