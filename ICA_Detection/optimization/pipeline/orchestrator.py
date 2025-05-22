# optimization/pipeline/orchestrator.py

from multiprocessing import Manager
from optimization import LOGGER
from optimization.cfgs.config import load_config
from optimization.engine.hpo import BayesianHyperparameterOptimizer
from typing import Optional

def run_hpo(config_path: str) -> None:
    """
    Main entrypoint: load config, set up resources, and run HPO.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.
    """
    cfg = load_config(config_path)
    LOGGER.info("Configuration loaded.")
    manager = Manager()
    gpu_lock = manager.Lock()
    available_gpus = manager.list(range(cfg.get("num_gpus", 1)))
    optimizer = BayesianHyperparameterOptimizer(
        config=cfg,
        logger=LOGGER,
        gpu_lock=gpu_lock,
        available_gpus=list(available_gpus)
    )
    optimizer.optimize()
