from __future__ import annotations
import os, logging
from typing import Optional
from ICA_Detection.optimization import LOGGER

def acquire_gpu(self) -> Optional[int]:
    """Return a *physical* GPU id, or None if none free."""
    with self.gpu_lock:                      # atomic
        if len(self.available_gpus) == 0:
            LOGGER.warning("No GPU free → running on CPU.")
            return None
        gpu_id = self.available_gpus[0]      # ListProxy always supports __getitem__
        del self.available_gpus[0]

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    torch.cuda.set_device(0)                 # logical id 0 after masking
    LOGGER.info("[Acquire GPU]: CUDA_VISIBLE_DEVICES=%s  torch sees %d device(s)",
             os.environ["CUDA_VISIBLE_DEVICES"],
             torch.cuda.device_count())

    LOGGER.info("[Acquire GPU]: Using physical GPU %d (logical 0).", gpu_id)
    return gpu_id

def release_gpu(self, gpu_id: Optional[int]) -> None:
    if gpu_id is None:
        return
    with self.gpu_lock:
        self.available_gpus.append(gpu_id)
    LOGGER.info("[Acquire GPU]: Released GPU %d.", gpu_id)