import sys
import logging
import argparse
from pathlib import Path

from ICA_Detection.tasks.mga_yolo.cfg.defaults import MaskGuidedAttentionConfig
from ICA_Detection.tasks.mga_yolo.engine.trainer import MaskGuidedTrainer


def setup_logging():
    """Configure logging for the MGA training process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("mga_training.log")],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MGA-YOLO Training")
    parser.add_argument(
        "--config",
        type=str,
        default="ICA_Detection/tasks/mga_yolo/assets/mga_config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--create-template",
        action="store_true",
        help="Create a template configuration file",
    )
    parser.add_argument(
        "--template-path",
        type=str,
        default="mga_config_template.yaml",
        help="Path for the template configuration file",
    )
    return parser.parse_args()


def main():
    """Main entry point for MGA training."""
    setup_logging()
    args = parse_args()

    # Create a template configuration if requested
    if args.create_template:
        MaskGuidedAttentionConfig.create_template(args.template_path)
        print(f"Created template configuration at: {args.template_path}")
        return

    # Check if configuration file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {args.config}")
        print("Use --create-template to create a template configuration file.")
        return

    # Load configuration from YAML
    config = MaskGuidedAttentionConfig.from_yaml(args.config)
    print(f"Loaded configuration from: {args.config}")

    # Display key configuration values
    print("\nTraining Configuration:")
    print(f"  Model: {config.model_cfg}")
    print(f"  Dataset: {config.data_yaml}")
    print(f"  Masks: {config.masks_folder}")
    print(f"  Target Layers: {config.target_layers}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Image Size: {config.imgsz}")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Device: {config.device}")

    # Initialize and run training
    trainer = MaskGuidedTrainer(config)
    trained_model = trainer.train()

    print(f"\nModel saved to {trained_model.ckpt_path}")


if __name__ == "__main__":
    main()
