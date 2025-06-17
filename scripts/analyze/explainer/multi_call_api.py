#!/usr/bin/env python
"""
Run `optimization_gradcam.py` on *several* images without any CLI flags.

How it works
------------
1.  We import the existing optimisation-GradCAM script.
2.  For every image name in `IMAGE_NAMES` we create an `argparse.Namespace`
    containing the exact same attributes the script expects.
3.  We monkey-patch `optimization_gradcam.get_args()` to return that namespace,
    call `optimization_gradcam.main()`, and restore the original function.

Nothing fancy – just a practical wrapper.
"""
from pathlib import Path
import types

import optimization_gradcam as opt  # ← the script you already have


# ------------------------------------------------------------------ parameters
ROOT_DIR   = Path("/media/mpascual/PortableSSD/Coronariografías/CompBioMed/bho_compbiomed/cadica/kfold/kfold_results/dca_yolo")            # ← change me
IMAGES_DIR = Path("/home/mpascual/research/datasets/angio/tasks/stenosis_detection/images")                   # ← change me
LABELS_DIR = Path("/home/mpascual/research/datasets/angio/tasks/stenosis_detection/labels/yolo")                   # ← change me

IMAGE_NAMES = [
    "cadica_p31_v12_00045"
]

ADAPTER      = "dca_yolov8"         # ultralytics or "dca_yolov8"
OUT_DIR      = Path("/media/mpascual/PortableSSD/Coronariografías/CompBioMed/bho_compbiomed/cadica/figures/gradcam")
CONF_THRESH  = 0.25                  # float
ADD_CBAR     = True                  # bool
ADD_LEGEND   = False                  # bool
CAM_METHOD   = "eigencam"            # e.g. "eigencam", "gradcam"
ADD_OPT_STR = False                  # bool

# -------------------------------------------------------------------- runner
def _run_one(stem: str):
    """Call optimization_gradcam.main() for a single image stem."""
    # Build the namespace that get_args() would normally return
    ns = types.SimpleNamespace(
        root        = ROOT_DIR,
        images      = IMAGES_DIR,
        labels      = LABELS_DIR,
        name        = stem,
        adapter     = ADAPTER,
        out         = OUT_DIR,
        conf_thres  = CONF_THRESH,
        add_cbar    = ADD_CBAR,
        cam_method  = CAM_METHOD,
        add_legend  = ADD_LEGEND,
        add_opt_str = ADD_OPT_STR
    )

    # Monkey-patch get_args(), run main(), restore original
    original_get_args = opt.get_args
    opt.get_args = lambda: ns
    try:
        opt.main()
    finally:
        opt.get_args = original_get_args


# -------------------------------------------------------------------- script
if __name__ == "__main__":
    for img_stem in IMAGE_NAMES:
        _run_one(img_stem)
