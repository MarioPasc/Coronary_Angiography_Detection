# optimization/engine/hpo.py

# optimization/engine/hpo.py
from typing import List, Any, Dict
from multiprocessing.managers import ListProxy

import math

import optuna # Ensure optuna is imported
from ICA_Detection.optimization.cfg.config import BHOConfig
from ICA_Detection.optimization.engine.trainers import get_trainer
from ICA_Detection.optimization import LOGGER

class BayesianHyperparameterOptimizer:
    """
    Manages Optuna study: sampler, pruner, parallelism, and result saving/plots.
    """

    def __init__(
        self,
        config: BHOConfig,
        gpu_lock: Any,
        available_gpus: ListProxy,
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
        self.logger = LOGGER
        self.gpu_lock = gpu_lock
        self.available_gpus = available_gpus
        TrainerCls = get_trainer(config.model_source)  # Get the appropriate trainer class
        self.trainer = TrainerCls(config, gpu_lock, available_gpus)  # Initialize the trainer

        # Wrap the trainer's train method to ensure it never returns None
        def safe_train(trial: optuna.Trial) -> float:
            result = self.trainer.train(trial)
            if result is None:
                self.logger.warning(
                    f"Trial {trial.number} returned None. Reporting objective based on direction."
                )
                # For maximization, a very small number; for minimization, a very large number.
                return -float('inf') if self.config.direction.lower() == "maximize" else float('inf')
            return result

        self._safe_train = safe_train

    def _default_cma_kwargs(self) -> Dict[str, Any]:
        """
        Build a rule-of-thumb CMA-ES config based solely on
        self.config.hyperparameters (no distribution objects).
        """
        # total number of tunables
        d = len(self.config.hyperparameters)

        # detect any discrete params: categorical or int
        has_discrete = any(
            hp_cfg.type in ("categorical", "int")
            for hp_cfg in self.config.hyperparameters.values()
        )

        # classical λ = 4 + floor(3 * ln(d))
        popsize = 4 + math.floor(3 * math.log(max(1, d)))

        return {
            "seed": self.config.seed,               # reproducible
            "x0": None,                             # midpoint by default
            "sigma0": None,                         # range/6 internally
            "popsize": popsize,
            "n_startup_trials": popsize,            # bootstrap
            "restart_strategy": "bipop",            # robust multimodal
            "inc_popsize": 2,                       # double on restart
            "consider_pruned_trials": False,        # default for MedianPruner
            "use_separable_cma": (d > 50),          # speed for high-dim
            "with_margin": has_discrete,            # avoid collapse on ints
            "lr_adapt": False,                      # off unless noisy
            "source_trials": None,                  # no warm-start
        }

    # NOTE: We are going to leave this here, but the NSGA-II algorithm is for
    # multi-objective optimization. If we want to use it, we will need to create
    # a trainer and hpo version that can handle multiple objectives.
    def _default_nsgaii_kwargs(self) -> Dict[str, Any]:
        return dict(
            population_size = getattr(self.config, "population_size", 50),
            crossover_prob  = getattr(self.config, "crossover_prob", 0.9),
            seed            = self.config.seed,
        )


    def optimize(self) -> None:
        """
        Build study, run trials, then save CSV and plots.
        """
        log = self.logger
        log.info("Starting HPO run.")
        optuna.logging.set_verbosity(optuna.logging.INFO) # Or make this configurable

        # 1) Storage & sampler
        storage_url = f"sqlite:///{self.config.storage}"
        log.debug(f"Using storage {storage_url}")
        
        # Pruner (currently NopPruner, could be made configurable)
        pruner = optuna.pruners.NopPruner()

        # Store sampler constructors (classes)
        sampler_constructors: Dict[str, Any] = {
            "tpe":       optuna.samplers.TPESampler,
            "random":    optuna.samplers.RandomSampler,
            "gpsampler": optuna.samplers.GPSampler,
            "qmcsampler": optuna.samplers.QMCSampler,
            "cmaes":     optuna.samplers.CmaEsSampler,
            "nsgaii":    optuna.samplers.NSGAIISampler,   
        }

        
        sampler_name = self.config.sampler.lower()
        sampler_constructor = sampler_constructors.get(sampler_name)

        if sampler_constructor is None:
            log.error(f"Unknown sampler: '{self.config.sampler}'. Defaulting to TPESampler.")
            sampler_constructor = optuna.samplers.TPESampler 
            sampler_name = "tpe" 

        # Prepare arguments for the chosen sampler
        sampler_kwargs = {"seed": self.config.seed}

        if sampler_name == "tpe":
            sampler_kwargs.update(
                consider_prior=True,
                prior_weight=1.0,
                consider_magic_clip=True,
                consider_endpoints=True,
                n_startup_trials=self.config.startup_trials,
                n_ei_candidates=24, # Optuna's default, can be configured
                multivariate=True,
                group=True,
                warn_independent_sampling=True, # Default is True
                constant_liar=True,
            )
        elif sampler_name == "gpsampler":
            independent_sampler_for_gps = optuna.samplers.RandomSampler(seed=self.config.seed)
            sampler_kwargs.update(
                independent_sampler= independent_sampler_for_gps,
                n_startup_trials= self.config.startup_trials,
                deterministic_objective= False # Default, can be configured
            )
        elif sampler_name == "qmcsampler":
            independent_sampler_for_qmc = optuna.samplers.RandomSampler(seed=self.config.seed)
            sampler_kwargs.update(
                qmc_type= 'sobol', # Default, can be configured
                scramble= True,    # Default, can be configured
                independent_sampler= independent_sampler_for_qmc,
                warn_asynchronous_seeding= True, # Default is True
                warn_independent_sampling= True  # Default is True
            )
        elif sampler_name == "cmaes":
            sampler_kwargs.update(self._default_cma_kwargs())
        elif sampler_name == "nsgaii":
            sampler_kwargs.update(self._default_nsgaii_kwargs())
        try:
            sampler = sampler_constructor(**sampler_kwargs) 
        except TypeError as e:
            log.error(f"Error instantiating sampler '{sampler_name}' with kwargs {sampler_kwargs}: {e}")
            log.info("Falling back to default TPESampler due to instantiation error.")
            sampler = optuna.samplers.TPESampler(
                seed=self.config.seed, 
                n_startup_trials=self.config.startup_trials
            )

        study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            storage=storage_url,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
        # ──────────────────────────────────────────────────────────
        # 2)  Run trials (or remaining trials)        
        # ──────────────────────────────────────────────────────────

        finished_trials = sum(t.state.is_finished() for t in study.trials)
        desired_total   = self.config.n_trials          # after editing YAML / CLI
        remaining       = max(0, desired_total - finished_trials)

        if remaining == 0:
            log.info(f"Study already has {finished_trials} finished trial(s) "
                     f"≥ requested total {desired_total}. Nothing to do.")
            return

        # parallelism cannot exceed ‘remaining’
        n_parallel_jobs = len(self.available_gpus) or 1
        n_parallel_jobs = min(n_parallel_jobs, remaining)

        log.info(f"Resuming study: {finished_trials} finished, "
                 f"running {remaining} more to reach {desired_total}. "
                 f"Using up to {n_parallel_jobs} parallel job(s).")

        study.optimize(
            self._safe_train,
            n_trials=remaining,
            n_jobs=n_parallel_jobs,
        )

        # 3) Save results
        # Consider making output paths configurable via BHOConfig
        results_dir = self.config.output_folder # Use output_folder from config
        import os
        os.makedirs(results_dir, exist_ok=True)

        csv_path = os.path.join(results_dir, f"{self.config.study_name}_results.csv")

        df = study.trials_dataframe()
        df.to_csv(csv_path, index=False)
        log.info(f"Results saved to {csv_path}")

        if self.config.save_plots:
            try:
                import optuna.visualization as vis
                if vis.is_available():
                    plot_configs = {
                        "history": vis.plot_optimization_history,
                        "contour": vis.plot_contour, # May require param_names for some studies
                        "parallel": vis.plot_parallel_coordinate,
                        "importance": vis.plot_param_importances,
                    }
                    for plot_name, plot_func in plot_configs.items():
                        try:
                            fig = plot_func(study)
                            fig_path = os.path.join(results_dir, f"{self.config.study_name}_{plot_name}.png")
                            fig.write_image(fig_path)
                        except (TypeError, ValueError) as e_plot: # Some plots might not be suitable for all studies/params
                             log.warning(f"Could not generate plot '{plot_name}': {e_plot}")
                    log.info(f"Plots saved in {results_dir}")
                else:
                    log.warning("Optuna visualization is not available (plotly might be missing). Skipping plot generation.")
            except ImportError:
                log.warning("plotly is not installed. Skipping plot generation. Run 'pip install plotly'.")
            except Exception as e:
                log.error(f"Error generating plots: {e}")