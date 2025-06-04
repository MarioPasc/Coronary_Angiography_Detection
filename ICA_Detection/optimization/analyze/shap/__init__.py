"""
SHAP analysis sub-package.

Convenience re-exports so you can simply write

>>> from ICA_Detection.optimization.analysis.shap import (
...     build_dataset,
...     train_and_save_models,
... )

and use the high-level helpers in notebooks.
"""
from ICA_Detection.optimization.analyze.shap.data_ingestion import build_dataset
from ICA_Detection.optimization.analyze.shap.model_training import train_and_save_models
from ICA_Detection.optimization.analyze.shap.visualization import make_strategy_heatmaps


__all__ = [
    "build_dataset",
    "train_and_save_models",
    "make_strategy_heatmaps",
]