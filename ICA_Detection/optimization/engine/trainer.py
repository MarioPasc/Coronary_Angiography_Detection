# optimization/engine/trainer.py

import time
import logging
import traceback
import gc
from typing import Any, Dict, Optional, Tuple, List

import torch
import optuna
from ICA_Detection.external.ultralytics.ultralytics import YOLO

from ICA_Detection.optimization.cfg.config import BHOConfig
from ICA_Detection.optimization import LOGGER
from ICA_Detection.optimization.cfg.defaults import DEFAULT_PARAMS
from ICA_Detection.optimization.utils.seeding import set_seed
from ICA_Detection.optimization.utils.hyperparameters import (
    prepare_hyperparameters,
    extract_hyperparameters,
)
from ICA_Detection.optimization.metrics.pruning import create_pruning_callback  
from ICA_Detection.optimization.metrics.gpu_monitor import create_gpu_monitoring_callbacks  


class YOLOTrainer:
    """
    Wraps a single Optuna trial: assigns resources, samples hyperparameters,
    runs YOLO training, logs GPU/metrics, and returns the objective.
    """

    def __init__(
        self,
        config: BHOConfig,
        gpu_lock: Any,
        available_gpus: List[int],
        logger: logging.Logger = LOGGER,
    ) -> None:
        """
        Parameters
        ----------
        config : BHOConfig
            Parsed optimization config.
        gpu_lock : Lock
            Multiprocessing lock to serialize GPU assignment.
        available_gpus : List[int]
            Pool of free GPU indices.
        logger : logging.Logger
            Logger for all messages.
        """
        self.config = config
        self.gpu_lock = gpu_lock
        self.available_gpus = available_gpus
        self.logger = logger

        # Build raw hyperparameter config dict from dataclasses
        raw_hparams: Dict[str, Dict[str, Any]] = {
            name: {
                "type": hp.type,
                "low": hp.low,
                "high": hp.high,
                "choices": hp.choices,
            }
            for name, hp in config.hyperparameters.items()
        }
        # Prepare the search space for Optuna
        self.hyperparameter_space: Dict[Tuple[str, str], Dict[str, Any]] = \
            prepare_hyperparameters(raw_hparams)

    def train(self, trial: optuna.Trial) -> Optional[float]:
        """
        Execute one trial: sample hyperparameters, assign GPU, train, log metrics,
        and return final F1 score as objective (or None on failure).

        Parameters
        ----------
        trial : optuna.Trial
            The current trial.

        Returns
        -------
        Optional[float]
            The final F1 score, or None if an error occurred.
        """
        # Adapter to include trial number in logs
        log = logging.LoggerAdapter(self.logger, {"trial": trial.number})
        log.info("Starting trial.")

        # 1) Reproducibility
        set_seed(self.config.seed)
        log.debug(f"Using seed {self.config.seed}.")

        # 2) Sample hyperparameters
        try:
            hparams = extract_hyperparameters(trial, self.hyperparameter_space, LOGGER)
            log.info(f"Sampled hparams: {hparams}")
        except ValueError as e:
            log.error(f"Hyperparam sampling error: {e}")
            return None

        # 3) Merge with defaults
        params = {**DEFAULT_PARAMS, **hparams}
        start_time = time.time()

        # 4) Acquire GPU
        with self.gpu_lock:
            if self.available_gpus:
                gpu_id = self.available_gpus.pop(0)
                device = f"cuda:{gpu_id}"
                trial.set_user_attr("gpu_id", gpu_id)
                log.info(f"Assigned GPU {gpu_id}.")
            else:
                gpu_id = None
                device = "cpu"
                log.warning("No GPU available; falling back to CPU.")
        params["device"] = device

        # 5) Build callbacks
        pruning_cb = create_pruning_callback(trial, LOGGER)
        epoch_start_cb, epoch_end_cb, train_end_cb = \
            create_gpu_monitoring_callbacks(trial, log)

        # 6) Run training
        try:
            # Set YOLO kwargs
            params.update({
                "data": self.config.data,
                "epochs": self.config.epochs,
                "imgsz": self.config.img_size,
                "val": True,
                "verbose": False,
                "name": f"trial_{trial.number}",
            })

            # Initialize model
            model = YOLO(self.config.model)
            model.model.to(device) # type: ignore[assignment]

            # Attach callbacks
            model.add_callback("on_train_epoch_start", epoch_start_cb)
            model.add_callback("on_train_epoch_end", epoch_end_cb)
            model.add_callback("on_train_end", train_end_cb)
            model.add_callback("on_train_epoch_end", pruning_cb)

            # Memory snapshot before
            mem_before = (
                torch.cuda.memory_allocated(device)
                if device != "cpu" else 0
            )

            # Launch training
            model.train(**params)

            # Memory snapshot after
            mem_after = (
                torch.cuda.memory_allocated(device)
                if device != "cpu" else 0
            )

            # Collect final metrics
            m = model.metrics.box # type: ignore[assignment]
            precision, recall = m.mp, m.mr
            map50, map50_95 = m.map50, m.map
            f1 = (2 * precision * recall) / (precision + recall) \
                 if precision + recall > 0 else 0.0

            # Log and record to trial
            elapsed = time.time() - start_time
            attrs = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "mAP50": map50,
                "mAP50-95": map50_95,
                "memory_before": mem_before,
                "memory_after": mem_after,
                "execution_time": elapsed,
            }
            for k, v in attrs.items():
                trial.set_user_attr(k, v)
            log.info(f"Trial completed: F1={f1:.4f}, time={elapsed:.1f}s")

            # Report and return
            trial.report(f1, step=train_end_cb.__self__().last_epoch)  # uses last_epoch set in callback
            return f1

        except Exception as e:
            log.error(f"Training error: {e}")
            log.error(traceback.format_exc())
            return None

        finally:
            # Return GPU and clean up
            with self.gpu_lock:
                if gpu_id is not None:
                    self.available_gpus.append(gpu_id)
                    log.info(f"Released GPU {gpu_id}.")
            del model
            torch.cuda.empty_cache()
            gc.collect()
