# optimization/engine/ultralytics_es.py
"""
Evolutionary hyper-parameter tuner (Ultralytics `model.tune`) that now supports
*both* the stock YOLOv8 models and the DCA-YOLOv8 fork.

The orchestrator still chooses this tuner when `sampler: ultralytics_es`; which
model family is trained depends on `config.model_source`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# ---------------- dynamic model map ----------------------------------- #
from ICA_Detection.external.ultralytics.ultralytics import YOLO as ULTRA_YOLO
from ICA_Detection.external.DCA_YOLOv8.DCA_YOLOv8.ultralytics.models.yolo import (
    YOLO as DCA_YOLO,
)

_MODEL_MAP = {
    "ultralytics": ULTRA_YOLO,
    "dca": DCA_YOLO,
}

# ---------------------------------------------------------------------- #

from ICA_Detection.optimization import LOGGER
from ICA_Detection.optimization.cfg.config import BHOConfig
from ICA_Detection.optimization.cfg.defaults import DEFAULT_PARAMS
from ICA_Detection.optimization.utils.gpu import acquire_gpu, release_gpu

class UltralyticsESTuner:
    """Single-process evolutionary search using `model.tune()`."""

    # construction ------------------------------------------------------ #
    def __init__(
        self,
        config: BHOConfig,
        gpu_lock: Any,  # multiprocessing.Manager().Lock
        available_gpus: Any,
    ) -> None:
        self.cfg = config
        self.gpu_lock = gpu_lock
        self.available_gpus = available_gpus  # type: ignore[assignment]

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
            gpu_id = acquire_gpu(self)
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
            device=0 if gpu_id is not None else "cpu",
            project=str(Path(self.cfg.output_folder)),
            name="ultralytics_es",
            verbose=True,
            plots=self.cfg.save_plots,
            save=True,
        )

        # 2) search space
        space: Dict[str, Any] = {}
        for name, hp in self.cfg.hyperparameters.items():
            if hp.type in {"uniform", "loguniform"}:
                space[name] = (hp.low, hp.high)
            elif hp.type == "categorical":
                space[name] = hp.choices

        # 3) run tuner
        model = self.model_cls(self.cfg.model)
        LOGGER.info(
            "[Ultralytics Evolutionary Strategy] %s | iterations=%d  epochs=%d",
            self.cfg.model_source.upper(),
            self.cfg.n_trials,
            self.cfg.epochs,
        )
        t0 = time.time()
        LOGGER.info(f"Arguments: {args}")
        model.tune(iterations=self.cfg.n_trials, space=space, **args)
        LOGGER.info("Evolution finished in %.1f min.", (time.time() - t0) / 60)

        # 4) move artefacts
        tune_dir = Path(model.save_dir) / "tune"
        dest = Path(self.cfg.output_folder) / "ultralytics_es"
        dest.mkdir(parents=True, exist_ok=True)
        for item in tune_dir.glob("*"):
            item.rename(dest / item.name)
        LOGGER.info("Results moved to %s", dest)
