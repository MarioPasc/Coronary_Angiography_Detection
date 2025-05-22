# optimization/cli.py

import argparse
from optimization.pipeline.orchestrator import run_hpo
from optimization import LOGGER

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
    args = parser.parse_args()
    try:
        run_hpo(args.config)
    except Exception as e:
        LOGGER.error(f"Fatal error: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
