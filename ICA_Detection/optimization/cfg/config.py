# optimization/cfgs/config.py

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
import yaml
import os

@dataclass
class HyperparameterConfig:
    """Configuration for one hyperparameter search space."""
    type: str
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None

@dataclass
class BHOConfig:
    """
    Parsed contents of a BHO YAML file.

    Attributes
    ----------
    model : str
        Path to the YOLO model weights.
    data : str
        Path to the dataset YAML.
    epochs : int
        Number of training epochs per trial.
    img_size : int
        Image size for training.
    n_trials : int
        Number of Optuna trials.
    save_plots : bool
        Whether to save visualization plots.
    storage : str
        Optuna storage URL.
    study_name : str
        Name of the Optuna study.
    seed : int
        Random seed.
    sampler : str
        Which sampler to use ('gpsampler', 'tpe', 'random', …).
    hyperparameters : Dict[str, HyperparameterConfig]
        Mapping from hyperparameter name to its search configuration.
    """
    model: str
    data: str
    epochs: int
    img_size: int
    n_trials: int
    startup_trials: int
    save_plots: bool
    direction: str
    storage: str
    study_name: str
    seed: int
    sampler: str
    hyperparameters: Dict[str, HyperparameterConfig]
    output_folder: str 
    model_source: Literal["ultralytics", "dca"]  # NEW, default="ultralytics"

    @staticmethod
    def from_yaml(path: str) -> "BHOConfig":
        """
        Load and validate a BHOConfig from a YAML file.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        # Build HyperparameterConfig objects
        hparams = {}
        for name, cfg in raw.get("hyperparameters", {}).items():
            hparams[name] = HyperparameterConfig(
                type=cfg["type"],
                low=cfg.get("low"),
                high=cfg.get("high"),
                choices=cfg.get("choices")
            )

        return BHOConfig(
            model_source=raw.get("model_source", "ultralytics"),  # Add with default
            model=raw["model"],
            data=raw["data"],
            direction=raw["direction"],
            epochs=raw["epochs"],
            img_size=raw["img_size"],
            n_trials=raw["n_trials"],
            startup_trials=raw.get("startup_trials", 0),
            save_plots=raw["save_plots"],
            storage=raw["storage"],
            study_name=raw["study_name"],
            seed=raw["seed"],
            sampler=raw["sampler"],
            hyperparameters=hparams,
            output_folder=raw.get("output_folder", "optimization_results") # Add with default
        )
