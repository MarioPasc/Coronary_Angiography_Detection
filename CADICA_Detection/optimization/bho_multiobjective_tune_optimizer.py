import logging
import optuna
from CADICA_Detection.external.ultralytics.ultralytics import YOLO
from typing import Dict, Tuple, List, Any
import pandas as pd
import yaml
import os
import random
import numpy as np
import torch
import gc
import multiprocessing
import time
import traceback
import nvidia_smi
from datetime import datetime

###############################################################################
#                             LOGGING CONFIG                                  #
###############################################################################

class TrialNumberFilter(logging.Filter):
    """
    A custom logging filter that adds a 'trial_number' attribute to logging records.

    This filter checks if the 'trial_number' attribute is present in a logging record. 
    If the attribute is not present, it assigns a default value of 'N/A'. This ensures 
    that log messages can include the trial number, even if it hasn't been explicitly set.

    Methods
    -------
    filter(record)
        Checks if the 'trial_number' attribute is in the logging record. 
        If not, sets 'trial_number' to 'N/A' and returns True.
    """

    def filter(self, record):
        """
        Checks if the 'trial_number' attribute is present in the logging record. 
        If it is not present, sets 'trial_number' to 'N/A'.

        Parameters
        ----------
        record : logging.LogRecord
            The logging record to be filtered and modified.

        Returns
        -------
        bool
            Always returns True, indicating that the record should be logged.
        """
        if not hasattr(record, 'trial_number'):
            record.trial_number = 'N/A'
        return True

# Set up logging
logger = logging.getLogger("BHOYOLO")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - Trial %(trial_number)s - %(message)s')

# Create file handler
fh = logging.FileHandler('log_bho_process.log')
fh.setFormatter(formatter)
fh.addFilter(TrialNumberFilter())
logger.addHandler(fh)

###############################################################################
#                           GPU USAGE LOGGING                                 #
###############################################################################
class NvidiaGPUUsageLogger:
    """Class to manage logging GPU usage, including time and device, using pandas."""

    def __init__(self, trial_number):
        self.trial_number = trial_number
        self.csv_file = "gpu_usage_log.csv"  # Single CSV file for all trials
        self.data = []  # List to accumulate log data

    def log_metrics(self, epoch, trainer):
        """Logs GPU metrics, including device, to the accumulated data."""
        model_device = next(trainer.model.parameters()).device  # Get the device dynamically
        device_index = model_device.index if model_device != 'cpu' else None
        
        # Fetch GPU metrics only if the device is a GPU
        if device_index is not None:
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_index)
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            temperature = nvidia_smi.nvmlDeviceGetTemperature(handle, nvidia_smi.NVML_TEMPERATURE_GPU)
            nvidia_smi.nvmlShutdown()

            metrics = {
                "trial": self.trial_number,
                "device": f"cuda:{device_index}",
                "epoch": epoch,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Current time
                "memory_free": mem.free / 1024 ** 2,  # Memory in MB
                "memory_total": mem.total / 1024 ** 2,  # Memory in MB
                "gpu_utilization": util.gpu / 100.0,  # Utilization percentage
                "memory_utilization": util.memory / 100.0,  # Utilization percentage
                "temperature": temperature  # Temperature in Celsius
            }
        else:
            # If using CPU, log minimal metrics
            metrics = {
                "trial": self.trial_number,
                "device": "cpu",
                "epoch": epoch,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Current time
                "memory_free": None,
                "memory_total": None,
                "gpu_utilization": None,
                "memory_utilization": None,
                "temperature": None
            }

        self.data.append(metrics)  # Append the metrics to the list

    def save_to_csv(self):
        """Writes the accumulated metrics to a single CSV file using pandas."""
        df = pd.DataFrame(self.data)
        # Append to the CSV file or create it if it doesn't exist
        if not os.path.exists(self.csv_file):
            df.to_csv(self.csv_file, index=False)
        else:
            df.to_csv(self.csv_file, mode='a', header=False, index=False)


###############################################################################
#                             YOLO CALLBACKS                                  #
###############################################################################

def create_gpu_monitoring_callbacks(trial, logger):
    """
    Creates callbacks for monitoring and logging GPU usage, including device info.
    """
    gpu_logger = NvidiaGPUUsageLogger(trial.number)

    def on_train_epoch_start(trainer):
        """Called at the start of each training epoch."""
        gpu_logger.log_metrics(trainer.epoch + 1, trainer)
        logger.info(f"GPU metrics logged at epoch start for trial {trial.number}")

    def on_train_epoch_end(trainer):
        """Called at the end of each training epoch."""
        gpu_logger.log_metrics(trainer.epoch + 1, trainer)
        logger.info(f"GPU metrics logged at epoch end for trial {trial.number}")

    def on_train_end(trainer):
        """Called when the training ends."""
        gpu_logger.save_to_csv()
        logger.info(f"GPU metrics saved to CSV for trial {trial.number}")

    return on_train_epoch_start, on_train_epoch_end, on_train_end


###############################################################################
#                          DEFAULT YOLO PARAMS                                #
###############################################################################

DEFAULT_PARAMS = {
    'data': None,  # This will be set dynamically in the class
    'epochs': 100,
    'patience': 20,
    'batch': 16,  # Automatic batch size determination
    'imgsz': 640,
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

###############################################################################
#                          OPTUNA OPTIMIZATOR                                 #
###############################################################################

def set_seed(seed: int = None) -> None:
    """
    Description
    --------------------------
    Sets the seed for generating random numbers to ensure reproducibility across various libraries.
    Args
    --------------------------
    seed: int
        The seed value to use for random number generators.
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

class BHOYOLO_Multiobjective_Optimizer:
    """
    A class for performing Bayesian Hyperparameter Optimization (BHO) on YOLO models.

    This class uses Optuna to optimize hyperparameters for YOLO model training. It includes 
    methods for setting up the YOLO training environment, extracting hyperparameters, and 
    preparing for the optimization process. It also manages GPU resources efficiently and 
    provides an interface for saving results and generating visualizations.

    Attributes
    ----------
    storage : str
        The database storage URL used by Optuna for saving study results.
    model : str
        The path to the YOLO model to be used for training.
    hyperparameters_config : Dict[str, Any]
        The configuration dictionary containing hyperparameters to be optimized.
    hyperparameters : Dict[Tuple[str, str], Any]
        The prepared hyperparameters for Optuna optimization.
    yaml_path : str
        The path to the YOLO data configuration file.
    epochs : int
        The number of training epochs.
    img_size : int
        The size of the images used for training.
    results : List[Dict[str, Any]]
        A list to store results from each trial.
    n_trials : int
        The number of optimization trials to run.
    save_plots : bool
        Whether to save visualizations of the optimization process.
    num_cpus : int
        The number of CPU cores available for training.
    num_gpus : int
        The number of GPUs available for training.
    available_gpus : List[int]
        A list of available GPU indices for training.
    gpu_lock : multiprocessing.Lock
        A lock for managing GPU resource allocation among parallel trials.
    device : str
        The computational device ('cuda' or 'cpu') used for training.
    default_params : Dict[str, Any]
        The default YOLO training parameters.

    Methods
    -------
    __init__(config: Dict[str, Any], storage: str = 'sqlite:///optuna_study.db', gpu_lock=None, available_gpus=None)
        Initializes the BHOYOLO class with configuration settings and resource management.

    _prepare_hyperparameters() -> Dict[Tuple[str, str], Any]
        Prepares the hyperparameters for optimization from the configuration dictionary.

    _extract_hyperparameters(trial: optuna.Trial, logger) -> Dict[str, Any]
        Extracts hyperparameters from the search space using the Optuna trial object.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        gpu_lock=None,
        available_gpus=None
    ) -> None:
        """
        Initializes the Bayesian Hyperparameter Optimization for YOLO model training.

        Parameters
        ----------
        config : Dict[str, Any]
            The configuration dictionary containing settings for YOLO training and hyperparameter optimization.
        storage : str, optional
            The database storage URL used by Optuna for saving study results (default is 'sqlite:///optuna_study.db').
        gpu_lock : multiprocessing.Lock, optional
            A lock for managing GPU resource allocation among parallel trials (default is None).
        available_gpus : List[int], optional
            A list of available GPU indices for training (default is None).

        Notes
        -----
        Initializes default parameters, sets up GPU resource management, and prepares the
        YOLO model for hyperparameter optimization.
        """
        # Get storage from config
        storage_path = config.get('storage', 'optuna_study.db')
        # Ensure the storage URL is correctly formatted
        if not storage_path.startswith('sqlite:///'):
            self.storage = f'sqlite:///{storage_path}'
        else:
            self.storage = storage_path
        self.study_name = config.get('study_name', 'optuna_study_default_name')
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
        self.available_gpus = available_gpus if available_gpus is not None else list(range(self.num_gpus))
        self.gpu_lock = gpu_lock if gpu_lock is not None else multiprocessing.Lock()

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

        Returns
        -------
        Dict[Tuple[str, str], Any]
            A dictionary of hyperparameters with their respective types and configurations.

        Raises
        ------
        ValueError
            If an unknown hyperparameter type is encountered in the configuration.

        Notes
        -----
        This method iterates over the hyperparameters provided in the configuration and
        organizes them for Optuna's optimization process.
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

        Parameters
        ----------
        trial : optuna.Trial
            The current Optuna trial object used for suggesting hyperparameters.
        logger : logging.Logger
            The logger object for logging messages related to hyperparameter extraction.

        Returns
        -------
        Dict[str, Any]
            A dictionary of hyperparameters suggested by the Optuna trial.

        Raises
        ------
        ValueError
            If an unknown function type is encountered while suggesting hyperparameters.

        Notes
        -----
        This method uses the Optuna trial object to suggest hyperparameter values based
        on the configured search space. It supports different types of distributions
        (e.g., 'loguniform', 'uniform', 'int', 'categorical') and logs the extraction process.
        """
        final_hyperparam: Dict[str, Any] = {}
        logger.debug(f"Extracting hyperparameters for trial {trial.number}.")

        for (hyperparam, func_type), param_config in self.hyperparameters.items():
            low = param_config.get('low')
            high = param_config.get('high')
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

        logger.info(f"Suggested optimizer: {final_hyperparam.get('optimizer', 'Adam')}")
        logger.info(f"Suggested batch size: {final_hyperparam.get('batch', 16)}")
        return final_hyperparam

    def _train_model(self, trial: optuna.Trial) -> float:
        """
        Trains the YOLO model using hyperparameters suggested by the Optuna trial.

        Parameters
        ----------
        trial : optuna.Trial
            The current Optuna trial object, used for suggesting hyperparameters and reporting results.

        Returns
        -------
        float
            The mAP@50-95 metric, which is used as the objective value for hyperparameter optimization.

        Description
        -----------
        This method sets up the YOLO model training environment using the provided trial object from
        Optuna. It assigns a GPU or falls back to CPU if no GPU is available, extracts the
        hyperparameters for the trial, and initializes the YOLO model. Training is conducted using
        these parameters, and a custom pruning callback is used to prune unpromising trials based
        on validation performance.

        Throughout the training process, various metrics and memory usage details are logged. After
        training, the results are saved and the GPU is returned to the pool for use by other trials.
        The method also handles resource cleanup and raises any exceptions encountered during
        training for Optuna to handle.

        Raises
        ------
        Exception
            If any error occurs during the training process, it is logged and re-raised.

        Notes
        -----
        - The method ensures reproducibility by setting a unique seed for each trial.
        - The YOLO model is explicitly moved to the assigned GPU, and memory usage is recorded before
        and after training.
        - The method supports GPU-based training if CUDA is available and manages GPU resources
        using a lock to prevent conflicts in a multi-GPU setting.
        """
        # Get the logger
        logger = logging.getLogger("BHOYOLO")

        # Create a LoggerAdapter to inject 'trial_number' into log messages
        logger = logging.LoggerAdapter(logger, {'trial_number': trial.number})

        logger.info(f"Starting training for trial {trial.number}.")

        # Generate a unique seed
        seed = trial.number
        set_seed(seed)
        logger.info(f"Seed for trial {trial.number}: {seed}")

        # Extract hyperparameters for the trial
        hyperparams = self._extract_hyperparameters(trial, logger)
        logger.info(f"Hyperparameters for trial {trial.number}: {hyperparams}")

        # Merge with default_params
        params = {**self.default_params, **hyperparams}

        # Record the start time
        start_time = time.time()

        # Assign a GPU
        with self.gpu_lock:
            if self.available_gpus:
                gpu_id = self.available_gpus.pop(0)
                assigned_device = f'cuda:{gpu_id}'
                trial.set_user_attr('gpu_id', gpu_id)
                logger.info(f"Assigned GPU {gpu_id} to trial {trial.number}")
                params['device'] = assigned_device
            else:
                assigned_device = 'cpu'
                gpu_id = None
                params['device'] = 'cpu'
                logger.info(f"No GPUs available, trial {trial.number} will run on CPU")

        # Log the assigned device
        logger.info(f"Assigned device for trial {trial.number}: {assigned_device}")

        # Verify the device availability
        if assigned_device != 'cpu' and not torch.cuda.is_available():
            logger.error(f"CUDA is not available. Trial {trial.number} will run on CPU instead.")
            assigned_device = 'cpu'
            gpu_id = None

        try:
            # Update parameters with configuration settings
            params['data'] = self.yaml_path
            params['epochs'] = self.epochs
            params['imgsz'] = self.img_size
            params['val'] = True  # Ensure validation is performed
            params['verbose'] = False  # Reduce verbosity if needed

            # Set unique name for each trial
            params['name'] = f"trial_{trial.number}_training"

            # Initialize the YOLO model
            model = YOLO(self.model)
            model.model.to(assigned_device)  # Explicitly move model to the assigned GPU
            
            # GPU usage callbacks
            on_train_epoch_start, on_train_epoch_end, on_train_end = create_gpu_monitoring_callbacks(trial, logger)

            # Add the callbacks to the YOLO model
            model.add_callback("on_train_epoch_start", on_train_epoch_start)
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            model.add_callback("on_train_end", on_train_end)


            # Verify that model parameters are on the correct device
            model_device = next(model.model.parameters()).device
            logger.info(f"Model parameters are on device: {model_device}")

            # Record memory before training
            if assigned_device != 'cpu':
                memory_allocated_before = torch.cuda.memory_allocated(device=assigned_device)
                memory_reserved_before = torch.cuda.memory_reserved(device=assigned_device)
            else:
                logging.info("Model assigned to cpu")
                memory_allocated_before = 0
                memory_reserved_before = 0
            logger.info(f"Memory allocated before training for trial {trial.number}: {memory_allocated_before/ 1e6} MB")
            logger.info(f"Memory reserved before training for trial {trial.number}: {memory_reserved_before/ 1e6} MB")

            # Log the device used by the model
            logger.info(f"Model device for trial {trial.number}: {model.device}")

            # Verify that model parameters are on the correct device
            model_device = next(model.model.parameters()).device
            logger.info(f"Model parameters are on device (before training): {model_device}")

            # Train the model using the merged hyperparameters
            model.train(**params)

            # Check if the model's device has changed after training
            post_train_device = next(model.model.parameters()).device
            logger.info(f"Model parameters are on device (after training): {post_train_device}")


            # Access the actual batch size used
            actual_batch_size = model.trainer.batch_size
            logger.info(f"Actual batch size used for trial {trial.number}: {actual_batch_size}")

            # Retrieve relevant metrics after training completes
            metrics = model.metrics

            # Access metrics directly from attributes
            precision_b = metrics.box.mp     # Mean Precision
            recall_b = metrics.box.mr        # Mean Recall
            map_50_b = metrics.box.map50     # mAP@50
            map_50_95 = metrics.box.map      # mAP@50-95
            f1_score = (2*precision_b*recall_b) / (precision_b + recall_b) if (precision_b+recall_b) > 0 else 0.0

            # Record memory after training
            if assigned_device != 'cpu':
                memory_allocated_after = torch.cuda.memory_allocated(device=assigned_device)
                memory_reserved_after = torch.cuda.memory_reserved(device=assigned_device)
            else:
                logging.info("Model assigned to cpu")
                memory_allocated_after = 0
                memory_reserved_after = 0
            logger.info(f"Memory allocated after training for trial {trial.number}: {memory_allocated_after / 1e6} MB")
            logger.info(f"Memory reserved after training for trial {trial.number}: {memory_reserved_after/ 1e6} MB")

            # Calculate execution time
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"Trial {trial.number} completed in {elapsed_time:.2f} seconds")

            # Save metrics to the trial object
            trial.set_user_attr('precision', precision_b)
            trial.set_user_attr('recall', recall_b)
            trial.set_user_attr('f1_score', f1_score)
            trial.set_user_attr('mAP50', map_50_b)
            trial.set_user_attr('mAP50-95', map_50_95)
            trial.set_user_attr('batch_size', actual_batch_size)
            trial.set_user_attr('seed', seed)
            trial.set_user_attr('execution_time', elapsed_time)
            trial.set_user_attr('memory_allocated_before', memory_allocated_before)
            trial.set_user_attr('memory_reserved_before', memory_reserved_before)
            trial.set_user_attr('memory_allocated_after', memory_allocated_after)
            trial.set_user_attr('memory_reserved_after', memory_reserved_after)

            # Save the current hyperparameters and metrics in the results list
            trial_results = {
                **hyperparams,
                'precision': precision_b,
                'recall': recall_b,
                'f1_score':f1_score,
                'mAP50': map_50_b,
                'mAP50-95': map_50_95,
                'trial_number': trial.number,
                'seed': seed,
                'gpu_id': gpu_id if assigned_device != 'cpu' else 'cpu',
                'batch_size': actual_batch_size,
                'execution_time': elapsed_time,
                'memory_allocated_before': memory_allocated_before,
                'memory_reserved_before': memory_reserved_before,
                'memory_allocated_after': memory_allocated_after,
                'memory_reserved_after': memory_reserved_after
            }
            self.results.append(trial_results)

            logger.info(f"Trial {trial.number} completed with mAP50-95: {map_50_95}")
            return map_50_95, f1_score

        except Exception as e:
            # Log any other exceptions that occurred during training
            logger.error(f"An error occurred during training of trial {trial.number}: {e}")
            logger.error(traceback.format_exc())
            raise e  # Re-raise exception to be caught by Optuna

        finally:
            # Return the GPU to the pool
            with self.gpu_lock:
                if gpu_id is not None:
                    self.available_gpus.append(gpu_id)
                    logger.info(f"Returned GPU {gpu_id} to the pool")
            # Resource cleanup
            del model
            torch.cuda.empty_cache()
            gc.collect()


    def optimize(self) -> None:
        """
        Description
        --------------------------
        Starts the hyperparameter optimization process using Optuna.
        
        References
        --------------------------
        TPESampler:
            BERGSTRA, James, et al. Algorithms for hyper-parameter optimization. 
            Advances in neural information processing systems, 2011, vol. 24.
            https://proceedings.neurips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf

            BERGSTRA, James; YAMINS, Dan; COX, David D. Making a science of model search. 
            arXiv preprint arXiv:1209.5111, 2012.
            https://arxiv.org/abs/1209.5111
        """
        # Get the logger
        logger = logging.getLogger("BHOYOLO")
        logger.info("Starting hyperparameter optimization.")

        logger.info(f"CUDA available?: {torch.cuda.is_available()}")
        logger.info(f"Available GPUs: {self.available_gpus}")
        # Get the name of each GPU
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        logger.info("Current device (no optimization executed): %s", torch.cuda.current_device())

        # Set up Optuna SQLite storage
        storage = optuna.storages.RDBStorage(
            url=self.storage,
            engine_kwargs={"connect_args": {"timeout": 10}}
        )

        # Define the sampler
        sampler = optuna.samplers.TPESampler(
            consider_prior=True,
            prior_weight=1.0,
            consider_magic_clip=True,
            consider_endpoints=True,
            n_startup_trials=15,
            n_ei_candidates=64,
            multivariate=True,
            group=True,
            warn_independent_sampling=True,
            constant_liar=True,
            seed=42
        )

        study = optuna.create_study(
            study_name=self.study_name,
            directions=["maximize", "maximize"],
            storage=storage,
            load_if_exists=True,
            sampler=sampler
        )

        n_initial_trials: int = 1
        self.n_initial_trials = n_initial_trials
        n_parallel_trials: int = self.n_trials - n_initial_trials

        logger.info(f"Running {n_initial_trials} initial trials sequentially.")
        try:
            study.optimize(self._train_model,
                           n_trials=n_initial_trials,
                           n_jobs=1,
                           gc_after_trial=True)
        except Exception as e:
            logger.error(f"An error occurred during the initial sequential trials: {e}")
            logger.error(traceback.format_exc())
            raise e

        if n_parallel_trials > 0:
            n_jobs: int = min(self.num_gpus, n_parallel_trials)
            logger.info(f"Running {n_parallel_trials} trials in parallel with {n_jobs} jobs.")
            try:
                study.optimize(self._train_model,
                               n_trials=n_parallel_trials,
                               n_jobs=n_jobs,
                               gc_after_trial=True)
            except Exception as e:
                logger.error(f"An error occurred during the parallel trials: {e}")
                logger.error(traceback.format_exc())
                raise e
        else:
            logger.info("No parallel trials to run.")


        # Save results
        self._save_results_to_csv(study)

        if self.save_plots:
            self._generate_visualizations(study)

    def _save_results_to_csv(self, study: optuna.Study) -> None:
        """
        Saves the results of each trial, including hyperparameters and performance metrics, to a CSV file.

        Parameters
        ----------
        study : optuna.Study
            The Optuna study object that contains all the completed trial information.

        Description
        -----------
        This method uses the Optuna `trials_dataframe` method to convert the results of each trial
        into a pandas DataFrame. The DataFrame includes details such as the trial number, objective
        value, hyperparameters, trial state, and user-defined attributes. The results are then saved
        to a CSV file named 'hyperparameter_optimization_results.csv' for easy analysis and
        record-keeping.

        Notes
        -----
        - The CSV file is created in the current working directory.
        - The method logs a message indicating that the results have been saved successfully.
        """
        df: pd.DataFrame = study.trials_dataframe(attrs=('number', 'value', 'params', 'state', 'user_attrs'))
        df.to_csv('hyperparameter_optimization_results.csv', index=False)
        logger = logging.getLogger("BHOYOLO")
        logger.info("Results saved to hyperparameter_optimization_results.csv")

    def _generate_visualizations(self, study: optuna.Study) -> None:
        """
        Generates and saves visualization plots for the hyperparameter optimization process.

        Parameters
        ----------
        study : optuna.Study
            The Optuna study object containing the trial history and performance data.

        Description
        -----------
        This method creates several visualization plots using Optuna's visualization module to help
        analyze the hyperparameter optimization process. The following plots are generated and saved:
        
        - **Contour Plot**: Visualizes the objective value as a function of two hyperparameters.
        - **Parallel Coordinates Plot**: Shows the relationship between hyperparameters and the
        objective value.
        - **Hyperparameter Importance Plot**: Ranks hyperparameters by their influence on the
        objective value.
        - **Optimization History Plot**: Tracks the objective values over the course of the study.

        The plots are saved as both PNG and SVG files in the current working directory. If an error
        occurs during plot generation, it is logged, and the stack trace is recorded for debugging.

        Notes
        -----
        - The plots provide a visual understanding of how hyperparameters affect the model's performance.
        - This method uses Optuna's `visualization` module, which requires `plotly` as a dependency.
        - The method handles exceptions gracefully and logs any errors encountered.
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
            logger.error(traceback.format_exc())

def main():
    """
    The main entry point for running the Bayesian Hyperparameter Optimization for the YOLO model.

    Description
    -----------
    This function sets up and runs the hyperparameter optimization process using Optuna. It:
    1. Loads the configuration from a YAML file.
    2. Initializes a multiprocessing manager to handle shared GPU resources.
    3. Creates an instance of the `BHOYOLO` class with the configuration and GPU resource manager.
    4. Calls the `optimize` method of the `BHOYOLO` class to start the optimization process.

    The function also sets up logging to capture detailed information about the execution and
    handles any exceptions that occur during the process, logging them for debugging purposes.

    Raises
    ------
    Exception
        If an error occurs during the execution, it is logged and re-raised.
    
    Notes
    -----
    - The configuration file should be in YAML format and include all necessary parameters.
    - The function assumes that the script is being run in a Python environment with the required
      libraries installed.
    """
    try:
        # Read configuration from YAML file
        with open('./bho_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger = logging.getLogger("BHOYOLO")
        logger.info("Configuration file loaded successfully.")

        # Create a manager for shared variables
        manager = multiprocessing.Manager()
        gpu_lock = manager.Lock()
        available_gpus = manager.list(range(torch.cuda.device_count()))

        # Instantiate the BHOYOLO class, passing shared variables
        bho_yolo = BHOYOLO_Multiobjective_Optimizer(config, gpu_lock=gpu_lock, available_gpus=available_gpus)

        # Run optimization
        bho_yolo.optimize()

    except Exception as e:
        logger = logging.getLogger("BHOYOLO")
        logger.error(f"An error occurred during execution: {e}")
        logger.error(traceback.format_exc())
        raise e

if __name__ == "__main__":
    main()
