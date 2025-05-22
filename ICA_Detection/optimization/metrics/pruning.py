# optimization/metrics/pruning.py

import optuna
from logging import Logger
from typing import Callable, Any

def create_pruning_callback(
    trial: optuna.Trial,
    logger: Logger
) -> Callable[[Any], None]:
    """
    Return a callback to prune unpromising trials based on F1 score.

    Parameters
    ----------
    trial : optuna.Trial
    logger : Logger

    Returns
    -------
    Callable[[trainer], None]
        To be called at end of each epoch.
    """
    def _callback(trainer: Any) -> None:
        epoch = trainer.epoch + 1
        metrics = trainer.metrics
        p = metrics.get('metrics/precision(B)', 0.0)
        r = metrics.get('metrics/recall(B)', 0.0)
        f1 = (2 * p * r) / (p + r) if p + r > 0 else 0.0
        logger.info(f"Trial {trial.number} | Epoch {epoch} | F1={f1:.4f}")
        trial.report(f1, step=epoch)
        if trial.should_prune():
            logger.info(f"Pruning trial {trial.number} at epoch {epoch}.")
            raise optuna.exceptions.TrialPruned()
    return _callback
