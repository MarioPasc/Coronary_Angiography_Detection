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

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_pruning_callback(trial, logger):
    """
    Description
    --------------------------
    Creates a callback function for Optuna to prune unpromising trials during YOLO model training.

    Args
    --------------------------
    trial: optuna.trial.Trial
        The current Optuna trial object.
    logger: logging.Logger
        The logger object to log training information.

    Returns
    --------------------------
    callable
        A callback function to be called at the end of each training epoch.
    """
    def optuna_pruning_callback(trainer):
        # Get the current epoch
        epoch = trainer.epoch + 1

        # Access the validation metrics
        metrics = trainer.metrics
        map50_95 = metrics.get('metrics/mAP50-95(B)', None)
        logger.info(f"Epoch {epoch}, Trial {trial.number}, mAP50-95: {map50_95}")

        if map50_95 is not None:
            # Report the metric to Optuna
            trial.report(map50_95, step=epoch)

            # Check if the trial should be pruned
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch}.")
                raise optuna.exceptions.TrialPruned()
    return optuna_pruning_callback

# Global default parameters for YOLO training
DEFAULT_PARAMS = {
    'data': None,  # This will be set dynamically in the class
    'epochs': 100,
    'time': None,
    'patience': 100,
    'batch': -1,
    'imgsz': 512,
    'save': True,
    'save_period': -1,
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
    'iou': 0.7,  # Standardized IoU value
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
    'pose': 12.0,
    'kobj': 1.0,
    'dropout': 0.0,
    'val': True,
    # Augmentation parameters
    'augment': False,
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
    'close_mosaic': 0,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'erasing': 0.0,
    'crop_fraction': 0.0,
    'auto_augment': "",
    'bgr': 0.0,
}

def set_seed(seed: int = 42):
    """
    Description
    --------------------------
    Sets the seed for generating random numbers to ensure reproducibility across various libraries.

    Args
    --------------------------
    seed: int
        The seed value to use for random number generators.
    """
    # Set Python's built-in random module seed
    random.seed(seed)
    # Set NumPy random seed
    np.random.seed(seed)
    # Set PyTorch random seed for CPU
    torch.manual_seed(seed)
    # Set PyTorch random seed for all GPUs
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for CUDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BHOYOLO:
    def __init__(
        self,
        config: Dict[str, Any]
    ) -> None:
        """
        Description
        --------------------------
        Initializes the Bayesian Hyperparameter Optimization for YOLO model training.

        Args
        --------------------------
        config: Dict[str, Any]
            Configuration dictionary loaded from YAML file containing model parameters and hyperparameter search space.
        """
        logging.info("Initializing BHOYOLO class.")
        # Read configurations from the config dictionary
        try:
            self.model: str = config.get('model', './yolov8lt.pt')
        except Exception as e:
            logging.error(f"Model {self.model} not recognized.")
            raise ValueError(f"Model {self.model} not recognized.")

        # Store hyperparameters and training details
        self.hyperparameters_config: Dict[str, Any] = config.get('hyperparameters', {})
        # Prepare hyperparameters for optimization
        self.hyperparameters = self._prepare_hyperparameters()
        # Set data path and training parameters
        self.yaml_path: str = config.get('data', '')
        self.epochs: int = config.get('epochs', 100)
        self.img_size: int = config.get('img_size', 512)
        # Initialize results list to store trial results
        self.results: List[Dict[str, Any]] = []
        # Number of trials for optimization
        self.n_trials: int = config.get('n_trials', 50)
        # Flag to save plots after optimization
        self.save_plots: bool = config.get('save_plots', True)
        # Email configuration (if any)
        self.email_config: Dict[str, str] = config.get('email', {})

        # Resource configuration
        # Get the number of CPUs allocated
        self.num_cpus: int = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
        # Detect the number of GPUs available
        self.num_gpus: int = torch.cuda.device_count()

        # Log the GPUs assigned
        visible_devices: int = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        logging.info(f"Visible CUDA devices: {visible_devices}")
        logging.info(f"Number of GPUs available: {self.num_gpus}")

        # Set the device parameter based on GPU availability
        if self.num_gpus > 0:
            self.device = 'cuda'  # Use all available GPUs
        else:
            self.device = 'cpu'

        # Copy the default parameters and update device and workers
        self.default_params: Dict[str, Any] = DEFAULT_PARAMS.copy()
        self.default_params['device'] = self.device
        self.default_params['workers'] = min(4 * self.num_gpus, self.num_cpus)


    def _prepare_hyperparameters(self) -> Dict[Tuple[str, str], Any]:
        """
        Description
        --------------------------
        Prepares the hyperparameters for optimization from the configuration dictionary.

        Returns
        --------------------------
        Dict[Tuple[str, str], Any]
            A dictionary of hyperparameters and their search spaces.
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
                logging.error(f"Unknown hyperparameter type {param_type} for {param_name}")
                raise ValueError(f"Unknown hyperparameter type {param_type} for {param_name}")
        return hyperparameters

    def _extract_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Description
        --------------------------
        Extracts hyperparameters from the search space using the Optuna trial object.

        Args
        --------------------------
        trial: optuna.Trial
            The Optuna trial object for suggesting hyperparameter values.

        Returns
        --------------------------
        Dict[str, Any]
            A dictionary of hyperparameters with values suggested by the trial.
        """
        final_hyperparam: Dict[str, Any] = {}
        logging.debug(f"Extracting hyperparameters for trial {trial.number}.")

        # Loop over the hyperparameters and their function type
        for (hyperparam, func_type), param_config in self.hyperparameters.items():
            if func_type == "loguniform":
                # Suggest a log-uniform value within the specified range
                low = param_config['low']
                high = param_config['high']
                final_hyperparam[hyperparam] = trial.suggest_loguniform(hyperparam, low, high)
            elif func_type == "uniform":
                # Suggest a uniform value within the specified range
                low = param_config['low']
                high = param_config['high']
                final_hyperparam[hyperparam] = trial.suggest_uniform(hyperparam, low, high)
            elif func_type == "int":
                # Suggest an integer value within the specified range
                low = param_config['low']
                high = param_config['high']
                final_hyperparam[hyperparam] = trial.suggest_int(hyperparam, low, high)
            elif func_type == "categorical":
                # Suggest a categorical value from the provided choices
                choices = param_config['choices']
                final_hyperparam[hyperparam] = trial.suggest_categorical(hyperparam, choices)
            else:
                logging.error(f"Unknown function type {func_type} for hyperparameter {hyperparam}")
                raise ValueError(f"Unknown function type {func_type} for hyperparameter {hyperparam}")

        return final_hyperparam

    def _train_model(self, trial: optuna.Trial) -> float:
        """
        Description
        --------------------------
        Trains the YOLO model using hyperparameters suggested by the Optuna trial.

        Args
        --------------------------
        trial: optuna.Trial
            The Optuna trial object for the current hyperparameter suggestion.

        Returns
        --------------------------
        float
            The mAP50-95 metric from the trained model, used as the objective value.

        Throws
        --------------------------
        optuna.exceptions.TrialPruned
            If the trial is pruned based on the pruning callback.
        Exception
            If any other error occurs during training.
        """
        # Create a logger for this trial
        logger = logging.getLogger(f"trial_{trial.number}")
        logger.setLevel(logging.INFO)

        # Ensure that log directory exists
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)

        # Create a file handler for the logger
        log_filename = os.path.join(log_dir, f"trial_{trial.number}.log")
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.INFO)

        # Create a logging format and add the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Start training
        logger.info(f"Starting training for trial {trial.number}.")

        # Set random seeds for reproducibility
        set_seed()

        # Assign a specific GPU to this trial
        gpu_id = trial.number % self.num_gpus
        device = f'cuda:{gpu_id}'
        logger.info(f"Trial {trial.number} assigned to GPU {device}")

        try:
            # Extract hyperparameters for the trial
            hyperparams = self._extract_hyperparameters(trial)
            logger.info(f"Hyperparameters for trial {trial.number}: {hyperparams}")

            # Merge with default_params
            params = {**self.default_params, **hyperparams}
            # Update parameters with configuration settings
            params['data'] = self.yaml_path
            params['epochs'] = self.epochs
            params['imgsz'] = self.img_size
            params['device'] = device  # Use the assigned GPU
            params['val'] = True  # Ensure validation is performed
            params['verbose'] = False  # Reduce verbosity if needed

            # Set unique name for each trial
            params['name'] = f"trial_{trial.number}_training"

            # Initialize the YOLO model
            model = YOLO(self.model)

            # Create the pruning callback with access to the trial object
            pruning_callback = create_pruning_callback(trial, logger)
            
            # Add the custom pruning callback
            model.add_callback("on_fit_epoch_end", pruning_callback)

            # Train the model using the merged hyperparameters
            results = model.train(**params)

            # Retrieve relevant metrics after training completes
            # Note: If the trial was pruned, the following lines might not be executed
            precision_b = results.box.mp
            recall_b = results.box.mr
            map_50_b = results.box.map50
            map_50_95 = results.box.map
        
            # Save metrics to the trial object
            trial.set_user_attr('precision', precision_b)
            trial.set_user_attr('recall', recall_b)
            trial.set_user_attr('mAP50', map_50_b)
            trial.set_user_attr('mAP50-95', map_50_95)

            # Save the current hyperparameters and metrics in the results list
            trial_results = {
                **hyperparams,
                'precision': precision_b,
                'recall': recall_b,
                'mAP50': map_50_b,
                'mAP50-95': map_50_95,
                'trial_number': trial.number
            }
            self.results.append(trial_results)

            logger.info(f"Trial {trial.number} completed with mAP50-95: {map_50_95}")
            return map_50_95

        except optuna.exceptions.TrialPruned as e:
            # Log that the trial was pruned
            logger.info(f"Trial {trial.number} pruned.")
            raise e

        except Exception as e:
            # Log any other exceptions that occurred during training
            logger.error(f"An error occurred during training of trial {trial.number}: {e}")
            raise e  # Re-raise exception to be caught by Optuna

        finally:
            # Clean up handlers to prevent duplicate logs
            logger.handlers.clear()
            # Resource cleanup
            del model
            torch.cuda.empty_cache()
            gc.collect()

    def optimize(self, sampler: optuna.samplers.BaseSampler = None, pruner: optuna.pruners.BasePruner = None) -> None:
        """
        Description
        --------------------------
        Starts the hyperparameter optimization process using Optuna.

        References
        --------------------------
        SuccessiveHalvingPruner: 
            LI, Liam, et al. A system for massively parallel hyperparameter tuning. 
            Proceedings of Machine Learning and Systems, 2020, vol. 2, p. 230-246.
            https://arxiv.org/abs/1810.05934
        
        TPESampler:
            BERGSTRA, James, et al. Algorithms for hyper-parameter optimization. 
            Advances in neural information processing systems, 2011, vol. 24.
            https://proceedings.neurips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf

            BERGSTRA, James; YAMINS, Dan; COX, David D. Making a science of model search. 
            arXiv preprint arXiv:1209.5111, 2012.
            https://arxiv.org/abs/1209.5111
        """
        logging.info("Starting hyperparameter optimization.")

        # Set up Optuna storage
        storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage('optuna_study.json'))

        # Define the pruner for early stopping
        if pruner is None:
            pruner = optuna.pruners.SuccessiveHalvingPruner(
                min_resource=10,
                reduction_factor=3,
                min_early_stopping_rate=0
            )

        # Define the sampler
        if sampler is None:
            sampler = optuna.samplers.TPESampler(
                consider_prior=True,
                prior_weight=1.0,
                consider_magic_clip=True,
                consider_endpoints=True,
                n_startup_trials=3,  
                n_ei_candidates=24,   
                multivariate=True,    # Enable multivariate TPE
                group=True,           # Enable group decomposition with multivariate TPE
                warn_independent_sampling=True,
                constant_liar=True,   # Useful for parallel optimization
                seed=42               # For reproducibility
            )

        # Create or load an Optuna study
        study = optuna.create_study(
            direction="maximize",
            storage=storage,
            load_if_exists=True,
            pruner=pruner,
            sampler=sampler
        )

        # Set the number of initial sequential trials
        n_initial_trials: int = 3
        self.n_initial_trials: int  = n_initial_trials  # Make accessible in _train_model

        n_parallel_trials: int = self.n_trials - n_initial_trials

        # Run initial trials sequentially
        logging.info(f"Running {n_initial_trials} initial trials sequentially.")
        try:
            study.optimize(
                self._train_model,
                n_trials=n_initial_trials,
                n_jobs=1
            )
        except Exception as e:
            logging.error(f"An error occurred during the initial sequential trials: {e}")
            raise e

        # Run remaining trials in parallel
        if n_parallel_trials > 0:
            n_jobs: int = min(self.num_gpus, n_parallel_trials)
            logging.info(f"Running {n_parallel_trials} trials in parallel with {n_jobs} jobs.")
            try:
                study.optimize(
                    self._train_model,
                    n_trials=n_parallel_trials,
                    n_jobs=n_jobs
                )
            except Exception as e:
                logging.error(f"An error occurred during the parallel trials: {e}")
                raise e
        else:
            logging.info("No parallel trials to run.")

        # Save results and generate visualizations
        self._save_results_to_csv(study)

        logging.info(f"Best trial mAP50-95: {study.best_trial.value}")
        logging.info(f"Best hyperparameters: {study.best_trial.params}")

        study.trials_dataframe().to_csv("study_results.csv", index=False)
        logging.info("Study results saved to study_results.csv")

        if self.save_plots:
            self._generate_visualizations(study)


    def _save_results_to_csv(self, study: optuna.Study) -> None:
        """
        Description
        --------------------------
        Saves the results of each trial (hyperparameters and metrics) to a CSV file.

        Args
        --------------------------
        study: optuna.Study
            The Optuna study object containing all trial results.
        """
        df: pd.DataFrame = study.trials_dataframe(attrs=('number', 'value', 'params', 'state', 'user_attrs'))
        df.to_csv('hyperparameter_optimization_results.csv', index=False)
        logging.info("Results saved to hyperparameter_optimization_results.csv")


    def _generate_visualizations(self, study: optuna.Study) -> None:
        """
        Description
        --------------------------
        Generates and saves visualizations for the optimization process.

        Args
        --------------------------
        study: optuna.Study
            The Optuna study object.
        """
        logging.info("Generating visualization plots.")
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


            logging.info("Visualization plots saved successfully.")
        except Exception as e:
            logging.error(f"An error occurred while generating visualizations: {e}")

def main() -> None:
    """
    Description
    --------------------------
    Main function to run the hyperparameter optimization process.
    """
    try:
        # Read configuration from YAML file
        with open('./bho_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logging.info("Configuration file loaded successfully.")

        # Instantiate the BHOYOLO class
        bho_yolo = BHOYOLO(config)

        # Run optimization
        bho_yolo.optimize()
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        raise e  # Optionally re-raise or handle the exception

if __name__ == "__main__":
    main()
