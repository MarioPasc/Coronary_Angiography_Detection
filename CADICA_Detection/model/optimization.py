import numpy as np
import os
import glob
import shutil
import pandas as pd
import logging
from typing import Dict, List
import yaml
import external.ultralytics as ultralytics

# Define hyperparameter limits
HYPERPARAMETER_LIMITS = {
    'momentum': (0.4, 0.8),
    'lr0': (0.0, 1.0e-3),
    'lrf': (0.0, 1.0e-2),
    'warmup_epochs': (2.5, 4),
    'warmup_momentum': (0.0, 1.0),
    'box': (7.0, 9.0),
    'cls': (0.5, 0.8),
    'dfl': (0.9, 1.3)
}

# Configure logging for detailed execution tracking
logging.basicConfig(filename='hyperparameter_tuning.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def _adjust_hyperparameter(hyperparam_name: str, base_value: float, coef: float) -> float:
    """
    Adjusts the hyperparameter based on a Gaussian distribution around the base value,
    ensuring the final value is within specified limits.

    Args
        hyperparam_name (str): The hyperparameter name being adjusted.
        base_value (float): The base value (center) for adjustment.
        coef (float): Coefficient to control exploration-exploitation. Higher values increase randomness.

    Returns
        float: Adjusted hyperparameter value constrained by predefined limits.

    Raises
        ValueError: If hyperparameter limits are undefined in `HYPERPARAMETER_LIMITS`.
    """
    try:
        min_value, max_value = HYPERPARAMETER_LIMITS.get(hyperparam_name, (None, None))
        if min_value is None or max_value is None:
            raise ValueError(f"Limits for '{hyperparam_name}' are not defined.")

        std_dev = base_value * coef
        adjusted_value = np.random.normal(loc=base_value, scale=std_dev)
        adjusted_value = max(min_value, min(adjusted_value, max_value))

        logging.info(f"Adjusted {hyperparam_name}: {adjusted_value} (base: {base_value}, coef: {coef})")
        return adjusted_value

    except Exception as e:
        logging.error(f"Error adjusting hyperparameter '{hyperparam_name}': {e}")
        raise

class HyperparameterTuning:
    
    def __init__(self, output: str, model: str, config_yaml_path: str, yaml_params_path: str) -> None:
        """
        Initializes the HyperparameterTuning class with the provided paths and configurations.
        
        Args
            output (str): Path to save results.
            model (str): YOLO model path or name.
            config_yaml_path (str): YOLO configuration file path.
            yaml_params_path (str): Path to YAML file with hyperparameters.
        """
        try:
            self.output_folder = output
            self.model = ultralytics.YOLO(model=model)
            self.config = config_yaml_path
            self.default_params = self._load_yaml_to_dict(yaml_params_path)

            os.makedirs(self.output_folder, exist_ok=True)
            logging.info("Initialized HyperparameterTuning with output path: %s", self.output_folder)
        
        except Exception as e:
            logging.error("Error during HyperparameterTuning initialization: %s", e)
            raise

    def _load_yaml_to_dict(self, yaml_path: str) -> Dict:
        """
        Loads a YAML file into a dictionary, logging the process.

        Args
            yaml_path (str): Path to YAML file.

        Returns
            Dict: Dictionary representation of the YAML file.

        Raises
            Exception: If the file cannot be loaded.
        """
        try:
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
            logging.info("Loaded YAML file: %s", yaml_path)
            return data
        except Exception as e:
            logging.error("Failed to load YAML file '%s': %s", yaml_path, e)
            raise

    def _train(self, hyperparameters: Dict[str, float], run_name: str, epochs_per_iteration: int) -> None:
        """
        Trains the model with specific hyperparameters and performs validation.

        Args
            hyperparameters (Dict[str, float]): Hyperparameters for the run.
            run_name (str): Name of the run for logging.
            epochs_per_iteration (int): Number of epochs for each iteration.

        Raises
            Exception: If training fails.
        """
        try:
            params = {**self.default_params, **hyperparameters, 'save_period': 1, 'name': run_name, 'epochs': epochs_per_iteration}
            logging.info("Starting training for run '%s' with hyperparameters: %s", run_name, hyperparameters)
            self.model.train(**params)
            self.val(run_name, hyperparameters)
        
        except Exception as e:
            logging.error("Training error for run '%s': %s", run_name, e)
            print("An error occurred during training. Check the log file for details.")

    def _move_validation_logs(self, run_name: str) -> None:
        """
        Moves validation log folders into `validation_log` within the specified run directory.

        Args
            run_name (str): Name of the current run.

        Raises
            Exception: If moving validation logs fails.
        """
        try:
            detect_folder = "./runs/detect"
            run_folder = os.path.join(detect_folder, run_name)
            validation_log_folder = os.path.join(run_folder, 'validation_log')

            os.makedirs(validation_log_folder, exist_ok=True)
            
            folders = [folder for folder in os.listdir(detect_folder) if folder.startswith('val')]
            for val_folder in folders:
                val_folder_path = os.path.join(detect_folder, val_folder)
                if os.path.isdir(val_folder_path):
                    shutil.move(val_folder_path, validation_log_folder)
                    logging.info("Moved %s to %s", val_folder, validation_log_folder)

        except Exception as e:
            logging.error("Error moving validation logs for run '%s': %s", run_name, e)

    def val(self, run_name: str, hyperparameters: Dict[str, float], tuning_csv: str = 'tuning_results.csv') -> None:
        """
        Validates the model, saves validation results for each hyperparameter set, and logs metrics.

        Args
            run_name (str): Name of the current run.
            hyperparameters (Dict[str, float]): Hyperparameters for validation.
            tuning_csv (str): Path to store tuning results CSV.

        Raises
            Exception: If validation fails.
        """
        try:
            weights_folder = f"./runs/detect/{run_name}/weights"
            weight_files = sorted([file for file in glob.glob(os.path.join(weights_folder, '*.pt')) if 'last.pt' not in file])
            weight_files.append(os.path.join(weights_folder, 'last.pt'))

            metrics_data = []
            best_weight_file = os.path.join(weights_folder, 'best.pt')
            validation_results_path = os.path.join(f"./runs/detect/{run_name}", "validation_results.csv")

            for weight_file in weight_files:
                model_batch = ultralytics.YOLO(weight_file)
                results = model_batch.val(imgsz=640, conf=0.001, plots=True, save_json=True)

                map_50 = results.box.map50
                map_50_95 = results.box.map
                file_name = os.path.basename(weight_file)

                metrics_data.append({
                    'File_name': file_name,
                    'map50_95': map_50_95,
                    'map50': map_50
                })

            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_csv(validation_results_path, mode='w', header=True, index=False)
            logging.info("Validation results saved to %s", validation_results_path)

            df_without_best = df_metrics[df_metrics['File_name'] != 'best.pt']
            map50_95_mean = df_without_best['map50_95'].mean()
            map50_95_last = df_metrics[df_metrics['File_name'] == 'last.pt']['map50_95'].values[0]
            map50_mean = df_without_best['map50'].mean()
            map50_last = df_metrics[df_metrics['File_name'] == 'last.pt']['map50'].values[0]

            final_metrics = {
                'Hyperparameter': list(hyperparameters.keys())[0],
                'Value': list(hyperparameters.values())[0],
                'map50_95_mean': map50_95_mean,
                'map50_95_last': map50_95_last,
                'map50_mean': map50_mean,
                'map50_last': map50_last
            }

            final_df = pd.DataFrame([final_metrics])
            final_df.to_csv(tuning_csv, mode='a', header=not os.path.exists(tuning_csv), index=False)
            logging.info("Tuning results for '%s' saved to %s", run_name, tuning_csv)

            self._move_validation_logs(run_name)

            if os.path.exists(best_weight_file):
                new_best_file_name = f'{list(hyperparameters.keys())[0]}_{run_name}_best.pt'
                shutil.copy(best_weight_file, os.path.join(self.output_folder, new_best_file_name))
                logging.info("Saved best.pt as '%s'", new_best_file_name)

            shutil.rmtree(weights_folder)
            logging.info("Removed weights folder: %s", weights_folder)

        except Exception as e:
            logging.error("Validation error for run '%s': %s", run_name, e)
            print("An error occurred during validation. Check the log file for details.")

    def tune_hyperparameters(self, hyperparameters: Dict[str, List[float]], coef: float = 0.1, num_iterations: int = 10, 
                             epochs_per_iteration: int = 5, random: bool = True) -> None:
        """
        Tunes hyperparameters by training the model with different configurations and logs each run.

        Args
            hyperparameters (Dict[str, List[float]]): Hyperparameters and ranges for tuning.
            coef (float): Exploration-exploitation coefficient for random tuning.
            num_iterations (int): Number of iterations for random tuning.
            epochs_per_iteration (int): Number of epochs for each run.
            random (bool): If True, uses random values; otherwise, uses predefined values.
        """
        for hyperparam_name, values in hyperparameters.items():
            base_value = self.default_params.get(hyperparam_name, None)
            if base_value is None:
                logging.warning("Hyperparameter '%s' not found in YAML parameters.", hyperparam_name)
                continue

            if random:
                logging.info("Random tuning for '%s' with base value %s.", hyperparam_name, base_value)
                for iteration in range(num_iterations):
                    run_name = f'{hyperparam_name}_run_{iteration}'
                    adjusted_value = _adjust_hyperparameter(hyperparam_name, base_value, coef)
                    logging.info("Random iteration %d: Adjusted %s to %s", iteration + 1, hyperparam_name, adjusted_value)
                    adjusted_hyperparameters = {hyperparam_name: adjusted_value}
                    self._train(adjusted_hyperparameters, run_name, epochs_per_iteration)
            else:
                logging.info("Predefined tuning for '%s'.", hyperparam_name)
                for iteration, value in enumerate(values):
                    run_name = f'{hyperparam_name}_run_{iteration}'
                    logging.info("Iteration %d: Using value %s for %s", iteration + 1, value, hyperparam_name)
                    provided_hyperparameters = {hyperparam_name: value}
                    self._train(provided_hyperparameters, run_name, epochs_per_iteration)

        logging.info("Tuning completed.")