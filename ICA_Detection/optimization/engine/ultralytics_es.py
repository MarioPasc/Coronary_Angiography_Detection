# optimization/engine/ultralytics_es.py
"""
Evolutionary hyper-parameter tuner (Ultralytics `model.tune`) that now supports
*both* the stock YOLOv8 models and the DCA-YOLOv8 fork.

The orchestrator still chooses this tuner when `sampler: ultralytics_es`; which
model family is trained depends on `config.model_source`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, List

import matplotlib
matplotlib.use("Agg")  # no GUI windows on headless servers


# ---------------- dynamic model map ----------------------------------- #
from ICA_Detection.external.ultralytics.ultralytics import YOLO as YOLO
from ICA_Detection.external.DCA_YOLOv8.DCA_YOLOv8.ultralytics.models.yolo import (
    YOLO as DCA_YOLO,
)

_MODEL_MAP = {
    "ultralytics": YOLO,
    "dca": DCA_YOLO,
}

# ---------------------------------------------------------------------- #

from ICA_Detection.optimization import LOGGER
from ICA_Detection.optimization.cfg.config import BHOConfig
from ICA_Detection.optimization.cfg.defaults import DEFAULT_PARAMS
from ICA_Detection.optimization.utils.gpu import acquire_gpu, release_gpu

# --------------------------------------------------------------------- #
# make sure "import ultralytics" works in *this* process *and* the
#     subprocess spawned by Tuner
# --------------------------------------------------------------------- #
def _ensure_ultralytics_importable(self) -> None:
    """
    If the top-level 'ultralytics' package is not installed, alias the vendored
    copy (ICA_Detection.external.ultralytics.ultralytics) under that name and
    prepend its root to PYTHONPATH so the child process finds it as well.
    """
    import importlib
    import sys
    import os
    from pathlib import Path


    if self.config.model_source == "ultralytics":
        vendored = importlib.import_module(
            "ICA_Detection.external.ultralytics.ultralytics"
        )

    elif self.config.model_source == "dca":
        vendored = importlib.import_module(
            "ICA_Detection.external.DCA_YOLOv8.DCA_YOLOv8.ultralytics"
        )

    # 1) alias in current interpreter
    sys.modules["ultralytics"] = vendored
    for sub in vendored.__all__ if hasattr(vendored, "__all__") else ():
        sys.modules[f"ultralytics.{sub}"] = getattr(vendored, sub, None)

    LOGGER.info(f"Aliased vendored 'ultralytics' to {vendored.__name__}.")

    # 2) add parent folder to PYTHONPATH for the subprocess
    root = Path(vendored.__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    os.environ["PYTHONPATH"] = (
        f"{root}{os.pathsep}" + os.environ.get("PYTHONPATH", "")
    )
    # -----------------------------------------------------------------
    # Torch ≥ 2.6 safeguard: allow Ultralytics checkpoints to unpickle
    # -----------------------------------------------------------------
    try:
        import torch
        from torch.serialization import add_safe_globals
        import ultralytics.nn.tasks as _tasks

        # (a)  allow the DetectionModel class when weights_only=True
        add_safe_globals([_tasks.DetectionModel])

        # (b)  monkey-patch Ultralytics' helper so it always forces
        #      weights_only=False (old behaviour) – tolerant to older Torch

        def _patched_torch_safe_load(path, map_location="cpu"):
            return torch.load(path, map_location=map_location,
                              weights_only=False), path

        _tasks.torch_safe_load = _patched_torch_safe_load
    except Exception as e:  # pragma: no cover
        LOGGER.warning("Torch checkpoint patching failed: %s", e)

def _patch_tuner(tuner, model) -> None:
    """
    Remove CLI keys from `tuner.args` and `tuner.space` that the target model
    family does not understand (derived from `model.cfg` or `.model.args`).

    Works for the DCA fork and falls through to a numeric-only whitelist for
    any future forks that lack both attributes.
    """
    # --- 1) build whitelist of banned keys ----------------------------
    banned: List[str] = ["bgr", "copy_paste_mode"]
    # --- 2) prune tuner.args ----------------------------------------
    bad = [k for k in vars(tuner.args) if k in banned]
    LOGGER.info(bad)
    for k in bad:
        tuner.args.__dict__.pop(k, None)



class UltralyticsESTuner:
    """Single-process evolutionary search using `model.tune()`."""

    # construction ------------------------------------------------------ #
    def __init__(
        self,
        config: BHOConfig,
        gpu_lock: Any,  # multiprocessing.Manager().Lock
        available_gpus: Any,
        selected_gpu: Optional[int] = None,
    ) -> None:
        self.cfg = config
        self.gpu_lock = gpu_lock
        self.available_gpus = available_gpus  # type: ignore[assignment]
        self.config = config
        self.selected_gpu = selected_gpu
        try:
            self.model_cls = _MODEL_MAP[config.model_source]
        except KeyError:  # pragma: no cover
            raise ValueError(
                f"Unsupported model_source '{config.model_source}'. "
                f"Choose from {list(_MODEL_MAP)}."
            ) from None

    # public API -------------------------------------------------------- #
    def optimize(self) -> None:
        gpu_id: Optional[int] = None
        try:
            if self.selected_gpu is None:
                gpu_id = acquire_gpu(self)
            else:
                gpu_id = self.selected_gpu
            self._tune(gpu_id)
        finally:
            release_gpu(self, gpu_id)

    # tuning core ------------------------------------------------------- #
    def _tune(self, gpu_id: Optional[int]) -> None:
        # 1) build YOLO args
        args: Dict[str, Any] = DEFAULT_PARAMS.copy()
        args.update(
            data=self.cfg.data,
            epochs=self.cfg.epochs,
            imgsz=self.cfg.img_size,
            device= f"cuda:{gpu_id}" if gpu_id is not None else "cpu",
            project=str(Path(self.cfg.output_folder)),
            name="ultralytics_es",
            verbose=True,
            plots=False, # hard-coded for incompatibility with research server
            save=True,
        )
        for k, v in list(args.items()):
            if v == "":
                args[k] = None

        # 2) search space
        space: Dict[str, Any] = {}
        for name, hp in self.cfg.hyperparameters.items():
            if hp.type in {"uniform", "loguniform"}:
                space[name] = (hp.low, hp.high)
            elif hp.type == "categorical":
                LOGGER.warning(
                    "[ES] Skipping categorical hyper-param '%s' – "
                    "Ultralytics tuner expects numeric ranges.", name
                )

        # 3) run tuner
        _ensure_ultralytics_importable(self=self)
        model = self.model_cls(self.cfg.model)
        LOGGER.info(
            "[Ultralytics Evolutionary Strategy] %s | iterations=%d  epochs=%d",
            self.cfg.model_source.upper(),
            self.cfg.n_trials,
            self.cfg.epochs,
        )
        t0 = time.time()
        LOGGER.info(f"Arguments before sanitization: {args}")

        from ultralytics.engine.tuner import Tuner

        tuner = Tuner(args=args, _callbacks=model.callbacks, )

        #_patch_tuner(tuner, model)
        LOGGER.info(f"Tunner args: {tuner.args}")

        model.tune(iterations=self.cfg.n_trials, space=space, **args)
        LOGGER.info("Evolution finished in %.1f min.", (time.time() - t0) / 60)

        # 4) move artefacts (only if tuning produced them)
        if hasattr(model, "save_dir") and (Path(model.save_dir) / "tune").exists():
            tune_dir = Path(model.save_dir) / "tune"
            dest = Path(self.cfg.output_folder) / "ultralytics_es"
            dest.mkdir(parents=True, exist_ok=True)
            for item in tune_dir.glob("*"):
                item.rename(dest / item.name)
            LOGGER.info("Results moved to %s", dest)
        else:
            LOGGER.warning("No tuning artefacts produced – "
                           "training probably failed upstream.")
