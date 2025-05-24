"""
Generic single-trial trainer for any YOLO-family model.

Sub-classes declare:
    MODEL_CLS       – constructor of the concrete model.
    DEFAULT_PARAMS  – baseline training kwargs for that family.

The rest (GPU booking, Optuna integration, metric logging, callbacks) is shared.
"""

from __future__ import annotations

import gc
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

import optuna
import torch

from multiprocessing.managers import ListProxy

from ICA_Detection.optimization import LOGGER
from ICA_Detection.optimization.cfg.config import BHOConfig
from ICA_Detection.optimization.metrics.pruning import create_pruning_callback
from ICA_Detection.optimization.metrics.gpu_monitor import (
    create_gpu_monitoring_callbacks,
)
from ICA_Detection.optimization.utils.seeding import set_seed
from ICA_Detection.optimization.utils.hyperparameters import (
    prepare_hyperparameters,
    extract_hyperparameters,
)
from ICA_Detection.optimization.engine.trainers._gpu import acquire_gpu, release_gpu  # expect self param


class BaseTrainer:
    """Runs one Optuna trial on one model and (optionally) one GPU."""

    # ---------- must be overridden by sub-classes ----------
    MODEL_CLS: ClassVar[Any] = None
    DEFAULT_PARAMS: ClassVar[Dict[str, Any]] = {}

    # -------------------------------------------------------
    def __init__(
        self,
        config: BHOConfig,
        gpu_lock: Any,
        available_gpus: ListProxy,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self.config = config
        self.gpu_lock = gpu_lock
        self.available_gpus = available_gpus
        self.logger = logger

        raw_hparams = {
            name: {
                "type": hp.type,
                "low": hp.low,
                "high": hp.high,
                "choices": hp.choices,
            }
            for name, hp in config.hyperparameters.items()
        }
        self.hyperparameter_space = prepare_hyperparameters(raw_hparams)

    # ------------------------------------------------------------------ #
    # --------------------  main entry called by hpo.py  ---------------- #
    # ------------------------------------------------------------------ #
    def train(self, trial: optuna.Trial) -> Optional[float]:
        log = logging.LoggerAdapter(self.logger, {"trial": trial.number})
        log.info("Starting trial.")

        # 1) reproducibility
        set_seed(self.config.seed)

        # 2) sample hyper-params
        try:
            hparams = extract_hyperparameters(
                trial, self.hyperparameter_space, self.logger
            )
            log.info(f"Sampled hparams: {hparams}")
        except ValueError as exc:
            log.error(f"Hyperparam sampling error: {exc}")
            return None

        # 3) merge with defaults
        params: Dict[str, Any] = {**self.DEFAULT_PARAMS, **hparams}

        # 4) book GPU
        gpu_id: Optional[int] = acquire_gpu(self)
        device: str = "cpu" if gpu_id is None else "cuda:0"
        params["device"] = device

        # 5) callbacks
        pruning_cb = create_pruning_callback(trial, self.logger)
        epoch_start_cb, epoch_end_cb, train_end_cb = create_gpu_monitoring_callbacks(
            trial, log, self.config.output_folder
        )

        # 6) populate mandatory YOLO args
        params.update(
            data=self.config.data,
            epochs=self.config.epochs,
            imgsz=self.config.img_size,
            project=os.path.join(self.config.output_folder, self.config.study_name),
            name=f"trial_{trial.number}",
            verbose=False,
            val=True,
        )

        start_time = time.time()

        try:
            # initialise model
            model = self.MODEL_CLS(self.config.model)
            self.logger.info(
                f"[Trainer] Instantiated model {self.MODEL_CLS}."
            )
            model.model.to(device)  # type: ignore[attr-defined]

            # attach callbacks
            model.add_callback("on_train_epoch_start", epoch_start_cb)
            model.add_callback("on_train_epoch_end", epoch_end_cb)
            model.add_callback("on_train_end", train_end_cb)
            model.add_callback("on_train_epoch_end", pruning_cb)

            mem_before = torch.cuda.memory_allocated(device) if device != "cpu" else 0
            model.train(**params)
            mem_after = torch.cuda.memory_allocated(device) if device != "cpu" else 0

            m = model.metrics.box  # type: ignore[attr-defined]
            precision, recall = m.mp, m.mr
            map50, map50_95 = m.map50, m.map
            f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0

            elapsed = time.time() - start_time
            user_attrs = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "mAP50": map50,
                "mAP50-95": map50_95,
                "memory_before": mem_before,
                "memory_after": mem_after,
                "execution_time": elapsed,
            }
            for k, v in user_attrs.items():
                trial.set_user_attr(k, v)

            trainer_obj = getattr(model, "trainer", None)
            last_epoch = getattr(trainer_obj, "last_epoch", getattr(trainer_obj, "epoch", -1))
            trial.report(f1, step=last_epoch)

            log.info(f"Trial completed: F1={f1:.4f}, time={elapsed:.1f}s")
            return f1

        except Exception as exc:  # pylint: disable=broad-except
            log.error(f"Training error: {exc}")
            log.error(traceback.format_exc())
            return None

        finally:
            release_gpu(self, gpu_id)
            del locals()["model"]  # make sure gc collects
            torch.cuda.empty_cache()
            gc.collect()
