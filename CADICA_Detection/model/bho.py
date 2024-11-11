import logging
import optuna
from ultralytics import YOLO
from typing import Dict, Tuple, List, Any
import pandas as pd
import yaml
import os
import random
import numpy as np
import torch
import gc
from threading import Lock

# Global lock for GPU assignment
gpu_lock = Lock()
available_gpus = []

# Function to set up logging for each trial
def setup_trial_logging(trial_number):
    logger = logging.getLogger(f"Trial_{trial_number}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"trial_{trial_number}.log")

    # Remove existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler for the trial
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

# Global default parameters for YOLO training
DEFAULT_PARAMS = {
    'data': None,  # This will be set dynamically in the class
    'epochs': 100,
    'batch': -1,  # Automatic batch size determination
    'imgsz': 512,
    'save': True,
    'cache': False,
    'device': None,
    'workers': 8,
    'project': None,
    'name': "default_name",
    'exist_ok': False,
    'pretrained': True,
    'verbose': True,
    'seed': 42,
    'deterministic': True,
    'single_cls': True,
    'rect': False,
    'cos_lr': True,
    'resume': False,
    'amp': True,
    'fraction': 1.0,
    'profile': False,
    'freeze': None,
    'plots': True,
    'optimizer': 'Adam',
    'iou': 0.7,
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'label_smoothing': 0.0,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'dropout': 0.0,
    'val': True,
    # Augmentation parameters
    'hsv_h': 0.0,
    'hsv_s': 0.0,
    'hsv_v': 0.0,
    'degrees': 0.0,
    'translate': 0.0,
    'scale': 0.0,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.0,
    'mosaic': 0.0,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'erasing': 0.0,
    'crop_fraction': 0.0,
    'auto_augment': "",
    'bgr': 0.0,
}

def set_seed(seed: int = None):
    """
    Sets the seed for generating random numbers to ensure reproducibility across various libraries.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for CUDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BHOYOLO:
    def __init__(
        self,
        config: Dict[str, Any],
        storage: str = 'sqlite:///optuna_study.db'
    ) -> None:
        """
        Initializes the Bayesian Hyperparameter Optimization for YOLO model training.
        """
        self.storage = storage
        self.model: str = config.get('model', './yolov8lt.pt')
        self.hyperparameters_config: Dict[str, Any] = config.get('hyperparameters', {})
        self.hyperparameters = self._prepare_hyperparameters()
        self.yaml_path: str = config.get('data', '')
        self.epochs: int = config.get('epochs', 100)
        self.img_size: int = config.get('img_size', 512)
        self.results: List[Dict[str, Any]] = []
        self.n_trials: int = config.get('n_trials', 50)
        self.save_plots: bool = config.get('save_plots', True)

        # Resource configuration
        self.num_cpus: int = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
        self.num_gpus: int = torch.cuda.device_count()
        self.available_gpus = list(range(self.num_gpus))

        # Set the device parameter based on GPU availability
        if torch.cuda.is_available() and self.num_gpus > 0:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Copy the default parameters and update device and workers
        self.default_params: Dict[str, Any] = DEFAULT_PARAMS.copy()
        self.default_params['device'] = self.device
        self.default_params['workers'] = min(4 * self.num_gpus, self.num_cpus)

    def _prepare_hyperparameters(self) -> Dict[Tuple[str, str], Any]:
        """
        Prepares the hyperparameters for optimization from the configuration dictionary.
        """
        hyperparameters: Dict[Tuple[str, str], Any] = {}
        # Iterate over the hyperparameters defined in the configuration
        for param_name, param_config in self.hyperparameters_config.items():
            param_type: str = param_config.get('type')
            # Check if the parameter type is valid
            if param_type in ['loguniform', 'uniform', 'int', 'categorical']:
                # Store the hyperparameter with its type and configuration
                hyperparameters[(param_name, param_type)] = param_config
            else:
                raise ValueError(f"Unknown hyperparameter type {param_type} for {param_name}")
        return hyperparameters

    def _extract_hyperparameters(self, trial: optuna.Trial, logger) -> Dict[str, Any]:
        """
        Extracts hyperparameters from the search space using the Optuna trial object.
        """
        final_hyperparam: Dict[str, Any] = {}
        logger.debug(f"Extracting hyperparameters for trial {trial.number}.")

        for (hyperparam, func_type), param_config in self.hyperparameters.items():
            low = param_config['low']
            high = param_config['high']
            if func_type == "loguniform":
                final_hyperparam[hyperparam] = trial.suggest_float(hyperparam, low, high, log=True)
            elif func_type == "uniform":
                final_hyperparam[hyperparam] = trial.suggest_float(hyperparam, low, high)
            elif func_type == "int":
                final_hyperparam[hyperparam] = trial.suggest_int(hyperparam, int(low), int(high))
            elif func_type == "categorical":
                choices = param_config['choices']
                final_hyperparam[hyperparam] = trial.suggest_categorical(hyperparam, choices)
            else:
                logger.error(f"Unknown function type {func_type} for hyperparameter {hyperparam}")
                raise ValueError(f"Unknown function type {func_type} for hyperparameter {hyperparam}")
        return final_hyperparam

    def _train_model(self, trial: optuna.Trial) -> float:
        """
        Trains the YOLO model using hyperparameters suggested by the Optuna trial.
        """
        # Set up a separate logger for this trial
        logger = setup_trial_logging(trial.number)

        # Start training
        logger.info(f"Starting training for trial {trial.number}.")

        # Generate a unique seed
        seed = trial.number
        set_seed(seed)
        logger.info(f"Seed for trial {trial.number}: {seed}")

        # Assign a GPU
        with gpu_lock:
            if self.available_gpus:
                gpu_id = self.available_gpus.pop(0)
                assigned_device = f'cuda:{gpu_id}'
                trial.set_user_attr('gpu_id', gpu_id)
                logger.info(f"Assigned GPU {gpu_id} to trial {trial.number}")
            else:
                assigned_device = 'cpu'
                logger.info(f"No GPUs available, trial {trial.number} will run on CPU")

        try:
            # Extract hyperparameters for the trial
            hyperparams = self._extract_hyperparameters(trial, logger)
            logger.info(f"Hyperparameters for trial {trial.number}: {hyperparams}")

            # Merge with default_params
            params = {**self.default_params, **hyperparams}
            # Update parameters with configuration settings
            params['data'] = self.yaml_path
            params['epochs'] = self.epochs
            params['imgsz'] = self.img_size
            params['val'] = True  # Ensure validation is performed
            params['verbose'] = False  # Reduce verbosity if needed
            params['device'] = assigned_device

            # Set unique name for each trial
            params['name'] = f"trial_{trial.number}_training"

            # Initialize the YOLO model
            model = YOLO(self.model)

            # Train the model using the merged hyperparameters
            results = model.train(**params)

            # Access the actual batch size used
            actual_batch_size = model.trainer.batch_size
            logger.info(f"Actual batch size used for trial {trial.number}: {actual_batch_size}")

            # Retrieve relevant metrics after training completes
            precision_b = results.box.mp
            recall_b = results.box.mr
            map_50_b = results.box.map50
            map_50_95 = results.box.map

            # Save metrics to the trial object
            trial.set_user_attr('precision', precision_b)
            trial.set_user_attr('recall', recall_b)
            trial.set_user_attr('mAP50', map_50_b)
            trial.set_user_attr('mAP50-95', map_50_95)
            trial.set_user_attr('batch_size', actual_batch_size)

            # Save the current hyperparameters and metrics in the results list
            trial_results = {
                **hyperparams,
                'precision': precision_b,
                'recall': recall_b,
                'mAP50': map_50_b,
                'mAP50-95': map_50_95,
                'trial_number': trial.number,
                'seed': seed,
                'gpu_id': gpu_id if assigned_device != 'cpu' else 'cpu',
                'batch_size': actual_batch_size,
            }
            self.results.append(trial_results)

            logger.info(f"Trial {trial.number} completed with mAP50-95: {map_50_95}")
            return map_50_95

        except Exception as e:
            # Log any other exceptions that occurred during training
            logger.error(f"An error occurred during training of trial {trial.number}: {e}")
            raise e  # Re-raise exception to be caught by Optuna

        finally:
            # Return the GPU to the pool
            with gpu_lock:
                if assigned_device != 'cpu':
                    self.available_gpus.append(gpu_id)
                    logger.info(f"Returned GPU {gpu_id} to the pool")
            # Resource cleanup
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
    def optimize(self, n_trials: int = None) -> None:
        """
        Starts the hyperparameter optimization process using Optuna.
        """
        # Set up a logger for the optimization process
        logger = logging.getLogger("BHOYOLO")
        logger.info("Starting hyperparameter optimization.")

        # Create or load an Optuna study
        study = optuna.create_study(
            study_name='shared-study',
            direction="maximize",
            storage=self.storage,
            load_if_exists=True,
        )

        # Run trials with threading
        if n_trials is None:
            n_trials = self.n_trials

        logger.info(f"Running {n_trials} trials with threading.")

        study.optimize(
            self._train_model,
            n_trials=n_trials,
            n_jobs=min(n_trials, self.num_gpus if self.num_gpus > 0 else 1),
        )

        # Save results
        self._save_results_to_csv(study)

        if self.save_plots:
            self._generate_visualizations(study)

    def _save_results_to_csv(self, study: optuna.Study) -> None:
        """
        Saves the results of each trial (hyperparameters and metrics) to a CSV file.
        """
        df: pd.DataFrame = study.trials_dataframe(attrs=('number', 'value', 'params', 'state', 'user_attrs'))
        df.to_csv('hyperparameter_optimization_results.csv', index=False)
        logger = logging.getLogger("BHOYOLO")
        logger.info("Results saved to hyperparameter_optimization_results.csv")

    def _generate_visualizations(self, study: optuna.Study) -> None:
        """
        Generates and saves visualizations for the optimization process.
        """
        logger = logging.getLogger("BHOYOLO")
        logger.info("Generating visualization plots.")
        try:
            import optuna.visualization as vis
            # Contour plot for 2D hyperparameter space exploration
            contour_fig = vis.plot_contour(study)
            contour_fig.write_image("contour_plot.png")
            contour_fig.write_image("contour_plot.svg")

            # Parallel coordinates plot to visualize hyperparameters and objective values
            parallel_fig = vis.plot_parallel_coordinate(study)
            parallel_fig.write_image("parallel_coordinate_plot.png")
            parallel_fig.write_image("parallel_coordinate_plot.svg")

            # Hyperparameter importance plot
            importance_fig = vis.plot_param_importances(study)
            importance_fig.write_image("hyperparameter_importance.png")
            importance_fig.write_image("hyperparameter_importance.svg")

            # Optimization history
            history = vis.plot_optimization_history(study)
            history.write_image("history_study.png")
            history.write_image("history_study.svg")

            logger.info("Visualization plots saved successfully.")
        except Exception as e:
            logger.error(f"An error occurred while generating visualizations: {e}")

def main():
    """
    Main function to run the hyperparameter optimization process.
    """
    try:
        # Read configuration from YAML file
        with open('./bho_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger = logging.getLogger("BHOYOLO")
        logger.info("Configuration file loaded successfully.")

        # Instantiate the BHOYOLO class
        bho_yolo = BHOYOLO(config)

        # Run optimization
        bho_yolo.optimize()

    except Exception as e:
        logger = logging.getLogger("BHOYOLO")
        logger.error(f"An error occurred during execution: {e}")
        raise e

if __name__ == "__main__":
    main()
