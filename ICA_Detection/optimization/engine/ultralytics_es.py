# optimization/engine/ultralytics_es.py
"""
Run Ultralytics' built-in evolutionary hyper-parameter tuner (`model.tune`).

The class mirrors the public interface of your Optuna optimizer so the
orchestrator can switch by checking `cfg.sampler`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from ICA_Detection.external.ultralytics.ultralytics import YOLO

from ICA_Detection.optimization.cfg.config import BHOConfig
from ICA_Detection.optimization.cfg.defaults import DEFAULT_PARAMS
from ICA_Detection.optimization import LOGGER


class UltralyticsESTuner:
    """Single-process evolutionary search using Ultralytics `YOLO.tune()`."""

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        config: BHOConfig,
        gpu_lock,               # multiprocessing.Manager().Lock
        available_gpus,         # multiprocessing.Manager().list
    ) -> None:
        self.cfg = config
        self.gpu_lock = gpu_lock
        self.available_gpus = available_gpus  # type: ignore[assignment]

    # ------------------------------------------------------------------ #
    # public entrypoint (called by orchestrator)
    # ------------------------------------------------------------------ #
    def optimize(self) -> None:
        gpu_id: Optional[int] = None
        try:
            gpu_id = self._acquire_gpu()
            self._tune(gpu_id)
        finally:
            self._release_gpu(gpu_id)

    # ------------------------------------------------------------------ #
    # gpu helpers
    # ------------------------------------------------------------------ #
    def _acquire_gpu(self) -> Optional[int]:
        """Return a *physical* GPU id, or None if none free."""
        with self.gpu_lock:                      # atomic
            if len(self.available_gpus) == 0:
                LOGGER.warning("No GPU free → running on CPU.")
                return None
            gpu_id = self.available_gpus[0]      # ListProxy always supports __getitem__
            del self.available_gpus[0]

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(0)                 # logical id 0 after masking
        LOGGER.info("Using physical GPU %d (logical 0).", gpu_id)
        return gpu_id

    def _release_gpu(self, gpu_id: Optional[int]) -> None:
        if gpu_id is None:
            return
        with self.gpu_lock:
            self.available_gpus.append(gpu_id)
        LOGGER.info("Released GPU %d.", gpu_id)

    # ------------------------------------------------------------------ #
    # tuning core
    # ------------------------------------------------------------------ #
    def _tune(self, gpu_id: Optional[int]) -> None:
        # ---- 1) build YOLO argument dict --------------------------------
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

        # ---- 2) derive search space -------------------------------------
        space: Dict[str, Any] = {}
        for name, hp in self.cfg.hyperparameters.items():
            if hp.type in {"uniform", "loguniform"}:
                space[name] = (hp.low, hp.high)
            elif hp.type == "categorical":
                space[name] = hp.choices

        # ---- 3) run tuner ----------------------------------------------
        model = YOLO(self.cfg.model)
        LOGGER.info(
            "Ultralytics ES: iterations=%d, epochs=%d", self.cfg.n_trials, self.cfg.epochs
        )
        t0 = time.time()
        model.tune(iterations=self.cfg.n_trials, space=space, optimizer="AdamW", **args)
        LOGGER.info("Evolution finished in %.1f min.", (time.time() - t0) / 60)

        # ---- 4) copy summary artefacts ----------------------------------
        tune_dir = Path(model.save_dir) / "tune"
        dest = Path(self.cfg.output_folder) / "ultralytics_es"
        dest.mkdir(parents=True, exist_ok=True)
        for item in tune_dir.glob("*"):
            item.rename(dest / item.name)
        LOGGER.info("Results moved to %s", dest)
