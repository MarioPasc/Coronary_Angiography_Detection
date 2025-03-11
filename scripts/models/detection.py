import os
import csv
import argparse
import torch
import yaml
from typing import Any, Dict

# from references/detection code
from ICA_Detection.external.torchvision.detection import (
    engine,
    utils,
    transforms as rtransforms,  # the reference/detection/transforms.py
    early_stopping,
)


# Model definitions
from ICA_Detection.models.detection.faster_rcnn import get_faster_rcnn_model
from ICA_Detection.models.detection.retinanet import get_retina_net_model
from ICA_Detection.models.detection.ssd import get_ssd_model

# Data loader function
from ICA_Detection.splits.holdout.holdout_detection_models import holdout_coco

#########################
# Helper functions
#########################


def get_transform():
    # If you only need "ToTensor", you can do:
    return rtransforms.Compose(
        [
            rtransforms.PILToTensor(),
            rtransforms.ToDtype(dtype=torch.float32, scale=True),
        ]
    )


def print_model_info(model: torch.nn.Module, config: argparse.Namespace) -> None:
    """
    Prints a summary of the model architecture, parameter counts,
    and the input configuration settings.

    Args:
        model (torch.nn.Module): The object detection model.
        config (argparse.Namespace): The configuration parameters (e.g., those read from a YAML or CLI).
    """
    # Print model summary
    print("\n==================== Model Summary ====================")
    print(model)

    # Compute and print total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n-------------------------------------------------------")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("-------------------------------------------------------\n")

    # Print input/configuration settings
    print("================ Input Configuration ==================")
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=======================================================\n")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Dictionary with configuration parameters.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def dict_to_namespace(config_dict: Dict[str, Any]) -> argparse.Namespace:
    """
    Converts a dictionary to an argparse.Namespace.

    Args:
        config_dict (Dict[str, Any]): The configuration dictionary.

    Returns:
        argparse.Namespace: A namespace object with attributes.
    """
    return argparse.Namespace(**config_dict)


###############################################################################
# Main training script
###############################################################################
def main(args):
    ############################################################################
    # Output directories
    ############################################################################
    model_dir = os.path.join(args.output_dir, args.model_type)
    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    csv_dir = os.path.join(model_dir, "csv")
    logs_dir = os.path.join(model_dir, "logs")
    results_dir = os.path.join(model_dir, "results")
    json_results_path = os.path.join(results_dir, "validation_preds.json")
    # For saving best model
    best_ckpt_dir = os.path.join(args.output_dir, "best_ckpt")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(best_ckpt_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, "training_log.csv")

    ############################################################################
    # Device
    ############################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ############################################################################
    # Data
    ############################################################################
    print("Loading dataset...")

    # If your transforms pipeline modifies both (img, target),
    # you can adapt it to the references transforms style.
    # If you only do "ToTensor", the below is enough:
    data_transforms = get_transform()

    train_loader, val_loader, test_loader = holdout_coco(
        coco_json_path=args.json_path,
        splits_info_path=args.splits_info_path,
        images_root=args.images_root,
        transforms=data_transforms,  # single or multi-arg transform
        batch_size=args.batch_size,
        shuffle_train=True,
    )

    print(f"Train set: {len(train_loader.dataset)} samples")
    print(f"Validation set: {len(val_loader.dataset)} samples")
    if test_loader:
        print(f"Test set: {len(test_loader.dataset)} samples")

    ############################################################################
    # Model
    ############################################################################
    print(f"Creating {args.model_type} model...")
    if args.model_type.lower() == "faster_rcnn":
        model = get_faster_rcnn_model(
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            freeze_until=args.freeze_until,
        )
    elif args.model_type.lower() == "retina_net":
        model = get_retina_net_model(
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            freeze_until=args.freeze_until,
        )
    elif args.model_type.lower() == "ssd":
        model = get_ssd_model(
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            freeze_until=args.freeze_until,
        )
    else:
        raise ValueError(
            f"Unsupported model type: {args.model_type}. Choose from 'faster_rcnn', 'retina_net', or 'ssd'"
        )

    model.to(device)

    print_model_info(model, args)

    ############################################################################
    # Optimizer and LR Scheduler
    ############################################################################
    optimizer = utils.get_optimizer(model, args)
    lr_scheduler = utils.get_lr_scheduler(optimizer, args)

    early_stopper = early_stopping.EarlyStopping(patience=args.patience, min_delta=0.0)
    # Track best checkpoint
    best_val_mAP_50_95 = 0.0
    ############################################################################
    # (Optional) Resume from Checkpoint
    ############################################################################
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    ############################################################################
    # CSV logging
    ############################################################################
    if not os.path.isfile(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_loss", "val_mAP_50_95", "val_mAP_50", "lr"]
            )

    ############################################################################
    # 3) Main Training Loop using references/detection "engine"
    ############################################################################
    print("Starting training...")
    num_epochs = args.epochs

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # -----------------------------------
        # Train one epoch
        # -----------------------------------
        # engine.train_one_epoch returns a dict of average losses
        train_stats = engine.train_one_epoch(
            model, optimizer, train_loader, device, epoch, print_freq=args.print_freq
        )
        # This call includes the standard warmup and logging logic if you want it.

        # For CosineAnnealingWarmRestarts, call scheduler.step() normally;
        # for ReduceLROnPlateau, you must call scheduler.step(val_loss)
        val_loss = engine.compute_validation_loss(model, val_loader, device)
        print(f"Validation loss for epoch {epoch+1}: {val_loss:.4f}")

        if args.cos_lr:
            lr_scheduler.step()
        else:
            lr_scheduler.step(val_loss)

        # -----------------------------------
        # Evaluate on val set
        # -----------------------------------
        # engine.evaluate returns a CocoEvaluator if it's recognized as COCO
        print("Evaluating on validation set...")
        coco_eval = engine.evaluate(
            model, val_loader, device=device, json_path=json_results_path, epoch=epoch
        )

        #
        if hasattr(coco_eval, "coco_eval") and "bbox" in coco_eval.coco_eval:
            stats = coco_eval.coco_eval["bbox"].stats
            val_mAP_50_95 = stats[0] * 100.0
            val_mAP_50 = stats[1] * 100.0

            # Save the best mAP@50-95
            if val_mAP_50_95 > best_val_mAP_50_95:
                print(
                    f"[INFO] Validation mAP improved from {best_val_mAP_50_95:.2f} to {val_mAP_50_95:.2f}."
                )
                best_val_mAP_50_95 = val_mAP_50_95
        else:
            # if for some reason it's not recognized as COCO
            val_mAP_50_95 = 0.0
            val_mAP_50 = 0.0

        # Extract the average training loss from train_stats if needed
        train_loss = (
            train_stats.meters["loss"].global_avg
            if "loss" in train_stats.meters
            else 0.0
        )
        current_lr = optimizer.param_groups[0]["lr"]

        # -----------------------------------
        # Log checkpoint and metrics
        # -----------------------------------
        # Write row to CSV
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    f"{train_loss:.4f}",
                    f"{val_mAP_50_95:.2f}",
                    f"{val_mAP_50:.2f}",
                    current_lr,
                ]
            )

        # Save checkpoint: if save_freq == -1, do not save intermediate checkpoints.
        if args.save_freq != -1:
            if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == num_epochs:
                ckpt_path = os.path.join(
                    checkpoints_dir, f"{args.model_type}_epoch_{epoch+1}.pth"
                )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    ckpt_path,
                )
                print(f"Checkpoint saved to {ckpt_path}")

        # -----------------------------------
        # Early stopping (based on val_loss)
        # -----------------------------------

        early_stopper.update(val_loss)
        if early_stopper.should_stop:
            print("[INFO] Early stopping triggered.")
            break

    # After training loop: Always save the final checkpoint.
    final_ckpt_path = os.path.join(
        checkpoints_dir, f"{args.model_type}_final_epoch_{epoch+1}.pth"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        final_ckpt_path,
    )
    print(f"Final checkpoint saved to {final_ckpt_path}")

    # -----------------------------
    # Final test if needed
    # -----------------------------
    if test_loader:
        print("\nEvaluating on test set...")

        engine.evaluate(
            model,
            test_loader,
            device=device,
            json_path=os.path.join(results_dir, "test_preds.json"),
            epoch="0",  # Placeholder for epoch
        )
        # If recognized as COCO, this prints final test AP

    print("\nTraining completed!")


###############################################################################
# 4) Argparse
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train object detection models using references/detection scripts"
    )
    # Only one argument: the path to the config file.
    parser.add_argument(
        "--config", type=str, default="args.yaml", help="Path to the YAML config file"
    )
    args_cli = parser.parse_args()

    # Load the YAML config and convert it to a Namespace.
    config_dict = load_config(args_cli.config)
    args = dict_to_namespace(config_dict)

    main(args)
