# optimization/utils/hyperparameters.py

from typing import Any, Dict, Tuple
import optuna
import logging

def prepare_hyperparameters(
    hyperparameters_config: Dict[str, Dict[str, Any]]
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Prepares the hyperparameters for optimization from the configuration dictionary.

    Parameters
    ----------
    hyperparameters_config : Dict[str, Dict[str, Any]]
        Raw 'hyperparameters' section from BHOConfig.

    Returns
    -------
    Dict[Tuple[str, str], Dict[str, Any]]
        Maps (param_name, param_type) to its config dict.

    Raises
    ------
    ValueError
        If an unknown hyperparameter type is encountered.
    """
    hyperparameters: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for name, cfg in hyperparameters_config.items():
        ptype = cfg.get('type')
        # Only these four types are supported
        if ptype in {'loguniform', 'uniform', 'int', 'categorical'}:
            hyperparameters[(name, ptype)] = cfg
        else:
            raise ValueError(f"Unknown hyperparameter type {ptype} for {name}")
    return hyperparameters

def extract_hyperparameters(
    trial: optuna.Trial,
    hyperparameters: Dict[Tuple[str, str], Dict[str, Any]],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Extracts hyperparameters from the search space using the Optuna trial object.

    Parameters
    ----------
    trial : optuna.Trial
        The current trial for suggesting hyperparameters.
    hyperparameters : Dict[Tuple[str, str], Dict[str, Any]]
        Output of prepare_hyperparameters.
    logger : logging.Logger
        Logger for debug/info messages.

    Returns
    -------
    Dict[str, Any]
        Suggested hyperparameter values for this trial.

    Raises
    ------
    ValueError
        If an unsupported search type is encountered.
    """
    final: Dict[str, Any] = {}
    logger.debug(f"Extracting hyperparameters for trial {trial.number}.")
    for (name, ptype), cfg in hyperparameters.items():
        low = cfg.get('low')
        high = cfg.get('high')
        if ptype == 'loguniform':
            final[name] = trial.suggest_float(name, low, high, log=True)
        elif ptype == 'uniform':
            final[name] = trial.suggest_float(name, low, high)
        elif ptype == 'int':
            final[name] = trial.suggest_int(name, int(low), int(high))
        elif ptype == 'categorical':
            final[name] = trial.suggest_categorical(name, cfg['choices'])
        else:
            logger.error(f"Unknown type {ptype} for {name}")
            raise ValueError(f"Unknown type {ptype} for hyperparameter {name}")

    # Log a couple of key suggestions
    logger.info(f"Suggested optimizer: {final.get('optimizer', 'Adam')}")
    logger.info(f"Suggested batch size: {final.get('batch', 16)}")
    return final
