import sys
import logging

from ICA_Detection.tasks.mga_yolo.cfg.defaults import MaskGuidedAttentionConfig
from ICA_Detection.tasks.mga_yolo.engine.trainer import MaskGuidedTrainer


def setup_logging():
    """Configure logging for the MGA training process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("mga_training.log")],
    )


def main():
    """Main entry point for MGA training."""
    setup_logging()

    # Parse command line arguments
    model_path = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.pt"

    # Create configuration
    config = MaskGuidedAttentionConfig(
        model_cfg=model_path,
        data_yaml="/home/mariopasc/Python/Datasets/COMBINED/YOLO_MGA/detection/yolo_ica_detection.yaml",
        masks_folder="/home/mariopasc/Python/Datasets/COMBINED/YOLO_MGA/masks",
        epochs=100,
        imgsz=512,
        project_dir="/home/mariopasc/Python/Datasets/COMBINED/detection/runs/train",
        experiment_name="mga_cbam_yolo",
    )

    # Initialize and run training
    trainer = MaskGuidedTrainer(config)
    trained_model = trainer.train()

    print(f"Model saved to {trained_model.ckpt_path}")


if __name__ == "__main__":
    main()
