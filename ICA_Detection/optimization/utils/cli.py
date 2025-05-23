# optimization/cli.py

import argparse
from ICA_Detection.optimization.pipeline.orchestrator import run_hpo
from ICA_Detection.optimization import LOGGER

def main() -> None:
    """
    Parse CLI arguments and invoke the HPO pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run Bayesian HPO for YOLO models."
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="YAML config file path."
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None, # Default to None, meaning use orchestrator's default
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not provided, defaults to [0]. If -1, we use all available GPUs."
    )
    args = parser.parse_args()
    try:
        run_hpo(config_path=args.config, gpu_ids_str=args.gpu_ids)
    except Exception as e:
        LOGGER.error(f"Fatal error: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
