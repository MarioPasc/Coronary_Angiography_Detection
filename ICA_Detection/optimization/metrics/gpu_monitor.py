# optimization/metrics/gpu_monitor.py

import os
import pandas as pd
import torch
from datetime import datetime
from typing import Any, List, Optional, Dict
import nvidia_smi
import logging
from ICA_Detection.optimization import LOGGER

class GPUUsageLogger:
    def __init__(self, trial_number: int, logger: Optional[logging.Logger] = None):
        self.trial_number = trial_number
        self.logger = logger or LOGGER
        self.nvml_initialized = False
        try:
            nvidia_smi.nvmlInit()
            self.nvml_initialized = True
        except nvidia_smi.NVMLError as e:
            self.logger.warning(f"Failed to initialize NVML for trial {self.trial_number}: {e}. GPU monitoring will be disabled.")

    def log(self, epoch: int, trainer: Any) -> None:
        if not self.nvml_initialized:
            return

        device_idx_to_monitor = None
        
        if hasattr(trainer, 'device') and isinstance(trainer.device, torch.device):
            device_obj = trainer.device
            if device_obj.type == 'cuda':
                if device_obj.index is not None:
                    device_idx_to_monitor = device_obj.index
                else:
                    self.logger.warning(
                        f"Trainer device is 'cuda' but its index is None (Trial {self.trial_number}, Epoch {epoch}). "
                        "GPU monitoring may be unreliable or default if not explicitly set."
                    )
                    # Fallback or error if necessary, for now, we'll let the None check below handle it.
            elif device_obj.type in ['cpu', 'mps']:
                # self.logger.debug(f"Trainer device is '{device_obj.type}'. Skipping NVML GPU monitoring for Trial {self.trial_number}, Epoch {epoch}.")
                return # Not a CUDA GPU, so skip NVML.
            else:
                 self.logger.warning(f"Trainer has an unrecognized device type: '{device_obj.type}'. Skipping NVML for Trial {self.trial_number}, Epoch {epoch}.")
                 return
        else:
            self.logger.error(
                f"Trainer object does not have a valid 'device' attribute of type torch.device. "
                f"Cannot determine device for GPU monitoring (Trial {self.trial_number}, Epoch {epoch})."
            )
            return

        if device_idx_to_monitor is None:
            self.logger.error(
                f"Could not determine a specific CUDA device index for GPU monitoring "
                f"(Trial {self.trial_number}, Epoch {epoch}). Trainer device info: {str(getattr(trainer, 'device', 'N/A'))}. Skipping NVML."
            )
            return

        try:
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_idx_to_monitor)
            # Assuming the rest of your NVML data fetching logic (utilization, memory) follows here.
            # For example:
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            self.logger.info(
                f"[GPU Monitor] Trial {self.trial_number} Epoch {epoch} - Device {device_idx_to_monitor}: "
                f"GPU Util: {util.gpu}%, Mem Used: {mem_info.used // (1024**2)}MB / {mem_info.total // (1024**2)}MB"
            )
        except nvidia_smi.NVMLError as e:
            self.logger.error(f"NVML error for device index {device_idx_to_monitor} (Trial {self.trial_number}, Epoch {epoch}): {e}")
        except TypeError as e: # This should ideally be caught by the None check above
            self.logger.critical(
                f"Internal TypeError when calling NVML (device_idx_to_monitor: {device_idx_to_monitor}, type: {type(device_idx_to_monitor)}): {e}. "
                "This indicates a bug in device index extraction."
            )
        except Exception as e:
            self.logger.error(f"Unexpected error during GPU monitoring for device index {device_idx_to_monitor} (Trial {self.trial_number}, Epoch {epoch}): {e}")

    def close(self) -> None:
        if self.nvml_initialized:
            try:
                nvidia_smi.nvmlShutdown()
            except nvidia_smi.NVMLError as e:
                self.logger.warning(f"Error during NVML shutdown for trial {self.trial_number}: {e}")
            self.nvml_initialized = False

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
        gpu_logger.close()
        logger.info(f"GPU monitoring closed for trial {trial.number}.")

    return on_train_epoch_start, on_train_epoch_end, on_train_end