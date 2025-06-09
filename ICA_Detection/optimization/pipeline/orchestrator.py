# optimization/pipeline/orchestrator.py
from __future__ import annotations

from typing import Optional, List, Union
import os 
from multiprocessing import Manager

from ICA_Detection.optimization import LOGGER
from ICA_Detection.optimization.cfg.config import BHOConfig
from ICA_Detection.optimization.engine.hpo import BayesianHyperparameterOptimizer
from ICA_Detection.optimization.engine.ultralytics_es import (
    UltralyticsESTuner,
)


def _parse_gpu_ids(gpu_ids_str: str | None) -> list[int]:
    """Convert CLI string to list[int]; -1 == detect from env."""
    if gpu_ids_str in (None, "", "-1"):
        env = (os.getenv("SLURM_JOB_GPUS") or
               os.getenv("CUDA_VISIBLE_DEVICES") or
               os.getenv("SLURM_STEP_GPUS") or "")
        if not env:
            return []                      # run on CPU
        return [int(x) for x in env.replace(" ", "").split(",")]
    if gpu_ids_str is None:
        return []
    return [int(x) for x in gpu_ids_str.split(",")] 


def run_hpo(config_path: str, gpu_ids_str: Optional[str] = None) -> None:
    """
    Main entrypoint: load config, set up resources, and run HPO.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.
    gpu_ids_str : Optional[str], optional
        Comma-separated string of GPU IDs, by default None.
    """
    cfg = BHOConfig.from_yaml(config_path)
    LOGGER.info("Configuration loaded.")

    gpus_to_use: List[int]
    
    if gpu_ids_str == "-1":
        # Detect available GPUs from the environment
        gpus_to_use = _parse_gpu_ids(gpu_ids_str)
        if not gpus_to_use:
            LOGGER.warning(f"Using all available GPUs: {gpus_to_use}")
    else:
        if gpu_ids_str:
            try:
                gpus_to_use = [int(gid.strip()) for gid in gpu_ids_str.split(',')]
                if not gpus_to_use: # Handle empty string after split if input was just ","
                    raise ValueError("GPU ID list cannot be empty if --gpu-ids is provided with non-empty value.")
                LOGGER.info(f"Using manually specified GPU IDs: {gpus_to_use}")
            except ValueError as e:
                LOGGER.error(f"Invalid GPU IDs format: '{gpu_ids_str}'. Error: {e}. Falling back to default [0].")
                gpus_to_use = [0]
        else:
            # Default behavior: use GPU 0.
            # You could extend this to read from cfg or detect all available GPUs if needed.
            gpus_to_use = [0] 
            LOGGER.info(f"No GPU IDs specified, defaulting to: {gpus_to_use}")

    manager = Manager()
    gpu_lock = manager.Lock()
    available_gpus = manager.list(gpus_to_use) # Use the determined list of GPUs

    LOGGER.info(f"Available GPUs: {available_gpus}")

    optimizer: Union[UltralyticsESTuner, BayesianHyperparameterOptimizer]
    if cfg.sampler.lower() == "ultralytics_es":
        optimizer = UltralyticsESTuner(
            config=cfg, gpu_lock=gpu_lock, available_gpus=available_gpus
        )
    else:
        optimizer = BayesianHyperparameterOptimizer(
            config=cfg, gpu_lock=gpu_lock, available_gpus=available_gpus
        )
    optimizer.optimize()
