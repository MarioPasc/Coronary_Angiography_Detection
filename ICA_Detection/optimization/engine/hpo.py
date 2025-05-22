# optimization/engine/hpo.py

import logging
from typing import List, Any, Dict

import optuna

from ICA_Detection.optimization.cfg.config import BHOConfig
from ICA_Detection.optimization.engine.trainer import YOLOTrainer
from ICA_Detection.optimization.pipeline.orchestrator import run_hpo  # for CLI entrypoint


class BayesianHyperparameterOptimizer:
    """
    Manages Optuna study: sampler, pruner, parallelism, and result saving/plots.
    """

    def __init__(
        self,
        config: BHOConfig,
        gpu_lock: Any,
        available_gpus: List[int],
    ) -> None:
        """
        Parameters
        ----------
        config : BHOConfig
            Parsed YAML config.
        gpu_lock : Lock
            For GPU assignment synchronization.
        available_gpus : List[int]
            Pool of GPU indices.
        """
        self.config = config
        self.logger = logging.getLogger("ica.optimization.optimizer")
        self.gpu_lock = gpu_lock
        self.available_gpus = available_gpus
        self.trainer = YOLOTrainer(
            config, gpu_lock, available_gpus, self.logger
        )

        # Wrap the trainer's train method to ensure it never returns None
        def safe_train(trial):
            result = self.trainer.train(trial)
            if result is None:
                self.logger.warning("Objective function returned None, replacing with -float('inf')")
                return float('-inf')
            return result

        self._safe_train = safe_train

    def optimize(self) -> None:
        """
        Build study, run initial sequential + parallel trials, then save CSV and plots.
        """
        log = self.logger
        log.info("Starting HPO run.")
        optuna.logging.set_verbosity(optuna.logging.INFO)

        # 1) Storage & sampler
        storage_url = f"sqlite:///{self.config.storage}"
        log.debug(f"Using storage {storage_url}")
        pruner = optuna.pruners.NopPruner()
        sampler_classes: Dict[str, Any] = {
            "tpe": optuna.samplers.TPESampler,
            "random": optuna.samplers.RandomSampler,
            "gpsampler": optuna.samplers.GPSampler,
            "qmcsampler": optuna.samplers.QMCSampler,
        }
        sampler_class = sampler_classes.get(self.config.sampler.lower())
        if sampler_class is None:
            raise ValueError(f"Unknown sampler: {self.config.sampler}")
        sampler = sampler_class(seed=self.config.seed)

        study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            storage=storage_url,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        # 2) Initial trial
        log.info("Running 1 initial trial sequentially.")
        study.optimize(
            self._safe_train,
            n_trials=1,
            n_jobs=1,
            gc_after_trial=True
        )
        remaining = self.config.n_trials - 1
        if remaining > 0:
            jobs = min(len(self.available_gpus), remaining) or 1
            log.info(f"Running {remaining} trials in parallel (jobs={jobs}).")
            study.optimize(
                self._safe_train,
                n_trials=remaining,
                n_jobs=jobs,
                gc_after_trial=True
            )

        # 4) Save results
        df = study.trials_dataframe(
            attrs=("number", "value", "params", "state", "user_attrs")
        )
        csv_path = "hyperparameter_optimization_results.csv"
        df.to_csv(csv_path, index=False)
        log.info(f"Results written to {csv_path}.")  # :contentReference[oaicite:2]{index=2}

        # 5) Optional plots
        if self.config.save_plots:
            try:
                import optuna.visualization as vis
                vis.plot_optimization_history(study).write_image("history.png")
                vis.plot_contour(study).write_image("contour.png")
                vis.plot_parallel_coordinate(study).write_image("parallel.png")
                vis.plot_param_importances(study).write_image("importance.png")
                log.info("Visualization plots saved.")  # :contentReference[oaicite:3]{index=3}
            except Exception as e:
                log.error(f"Plotting error: {e}")

