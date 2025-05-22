# optimization/metrics/gpu_monitor.py

import os
import pandas as pd
from datetime import datetime
from typing import Any, List, Optional, Dict
import nvidia_smi
from logging import Logger
from ICA_Detection.optimization import LOGGER

class GPUUsageLogger:
    """
    Accumulates and writes GPU usage metrics to CSV.
    """

    def __init__(self, trial_number: int, logger: Logger = LOGGER) -> None:
        """
        Parameters
        ----------
        trial_number : int
            Optuna trial number.
        logger : Logger
            Logger instance for internal messages.
        """
        self.trial = trial_number
        self.logger = logger
        self.csv_file = "gpu_usage_log.csv"
        self.records: List[Dict[str, Any]] = []

    def log(self, epoch: int, device_index: Optional[int]) -> None:
        """
        Fetch and store one snapshot of GPU metrics.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        device_index : Optional[int]
            GPU device index, or None for CPU.
        """
        timestamp = datetime.now().isoformat(sep=' ', timespec='seconds')
        if device_index is not None:
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_index)
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            temp = nvidia_smi.nvmlDeviceGetTemperature(handle, 0)
            nvidia_smi.nvmlShutdown()
            rec = {
                "device": device_index,
                "trial": self.trial,
                "epoch": epoch,
                "time": timestamp,
                "device": f"cuda:{device_index}",
                "gpu_util": util.gpu,
                "mem_free_MB": mem.free / 1024**2,
                "mem_total_MB": mem.total / 1024**2,
                "temp_C": temp,
            }
        else:
            rec = {
                "device": None,
                "trial": self.trial,
                "epoch": epoch,
                "time": timestamp,
                "device": "cpu",
                "gpu_util": None,
                "mem_free_MB": None,
                "mem_total_MB": None,
                "temp_C": None,
            }
        self.records.append(rec)

    def save(self) -> None:
        """
        Append all recorded metrics to the CSV file.
        """
        df = pd.DataFrame(self.records)
        header = not os.path.exists(self.csv_file)
        df.to_csv(self.csv_file, mode='a', header=header, index=False)
        self.logger.info("GPU usage log saved.")

def create_gpu_monitoring_callbacks(trial, logger):
    """
    Creates callbacks for monitoring GPU usage and tracking the current epoch dynamically.
    """
    gpu_logger = GPUUsageLogger(trial.number)  # Existing GPU logger
    current_epoch = {"epoch": 0}  # Dictionary to store the current epoch count (mutable)

    def on_train_epoch_start(trainer):
        """Called at the start of each training epoch."""
        current_epoch["epoch"] += 1  # Increment the epoch counter
        trainer.current_epoch = current_epoch["epoch"]  # Attach to the trainer object
        logger.info(f"Epoch {current_epoch['epoch']} started for trial {trial.number}.")
        gpu_logger.log(current_epoch["epoch"], trainer)

    def on_train_epoch_end(trainer):
        """Called at the end of each training epoch."""
        logger.info(f"Epoch {current_epoch['epoch']} ended for trial {trial.number}.")
        gpu_logger.log(current_epoch["epoch"], trainer)

    def on_train_end(trainer):
        """Called when the training ends."""
        logger.info(f"Training completed at epoch {current_epoch['epoch']} for trial {trial.number}.")
        trainer.last_epoch = current_epoch["epoch"]  # Attach the last epoch

    return on_train_epoch_start, on_train_epoch_end, on_train_end