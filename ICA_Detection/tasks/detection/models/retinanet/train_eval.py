import os
import csv
import argparse
import torch
import yaml
from typing import Any, Dict

# Your engine with train_one_epoch, evaluate, compute_validation_loss


from ICA_Detection.tasks.detection.utils import (
    utils,
    transforms as rtransforms,
    early_stopping,
    retina_wrapper,
)

# Retinanet model
from ICA_Detection.tasks.detection.models.retinanet.model import resnet152
from ICA_Detection.tasks.detection.models.retinanet import engine
from ICA_Detection.tasks.detection.models.retinanet.dataloader import (
    holdout_coco_retina,
)


#########################
# Helper functions
#########################
def get_transform():
    """
    Example transform: convert PIL -> tensor, scale to [0,1].
    If you have custom transforms, adapt them here.
    """
    return rtransforms.Compose(
        [
            rtransforms.PILToTensor(),  # (C,H,W) float
            rtransforms.ToDtype(dtype=torch.float32, scale=True),  # in [0,1]
        ]
    )


def print_model_info(model: torch.nn.Module, config: argparse.Namespace) -> None:
    """
    Prints a summary of the model architecture, parameter counts,
    and input configuration settings.
    """
    print("\n==================== Model Summary ====================")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n-------------------------------------------------------")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("-------------------------------------------------------\n")

    print("================ Input Configuration ==================")
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=======================================================\n")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def dict_to_namespace(config_dict: Dict[str, Any]) -> argparse.Namespace:
    """
    Converts a dictionary to an argparse.Namespace.
    """
    return argparse.Namespace(**config_dict)


###############################################################################
# Main training script
###############################################################################
def main(args):
    ###########################################################################
    # Output directories
    ###########################################################################
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

    ###########################################################################
    # Device
    ###########################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ###########################################################################
    # Data
    ###########################################################################
    print("Loading dataset...")

    data_transforms = get_transform()

    # Build train/val/test DataLoaders
    train_loader, val_loader, test_loader = holdout_coco_retina(
        coco_json_path=args.json_path,
        splits_info_path=args.splits_info_path,
        images_root=args.images_root,
        transform=data_transforms,
        batch_size=args.batch_size,
        shuffle_train=True,
    )

    # The collate_fn in holdout_coco_retina returns `batch` as a list of samples.
    # That is enough for our code as long as engine.train_one_epoch, etc.
    # parse it properly.

    print(f"Train set: {len(train_loader.dataset)} samples")
    print(f"Validation set: {len(val_loader.dataset)} samples")
    if test_loader:
        print(f"Test set: {len(test_loader.dataset)} samples")

    ###########################################################################
    # Model
    ###########################################################################
    print(f"Creating {args.model_type} model...")
    # Example: RetinaNet with resnet152 backbone
    if args.model_type.lower() == "retina_net":
        model = resnet152(num_classes=args.num_classes, pretrained=args.pretrained)
        model = retina_wrapper.RetinaNetWrapper(retinanet_model=model)
    else:
        raise ValueError(f"Only 'retina_net' supported in this example.")

    model.to(device)

    print_model_info(model, args)

    ###########################################################################
    # Optimizer and LR Scheduler
    ###########################################################################
    optimizer = utils.get_optimizer(model, args)
    lr_scheduler = utils.get_lr_scheduler(optimizer, args)

    early_stopper = early_stopping.EarlyStopping(patience=args.patience, min_delta=0.0)
    best_val_mAP_50_95 = 0.0

    ###########################################################################
    # (Optional) Resume from Checkpoint
    ###########################################################################
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    ###########################################################################
    # CSV logging
    ###########################################################################
    if not os.path.isfile(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_loss", "val_mAP_50_95", "val_mAP_50", "lr"]
            )

    ###########################################################################
    # Main Training Loop
    ###########################################################################
    print("Starting training...")
    num_epochs = args.epochs

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # 1) Train one epoch
        train_stats = engine.train_one_epoch(
            model, optimizer, train_loader, device, epoch, print_freq=args.print_freq
        )
        val_loss = engine.compute_validation_loss(model, val_loader, device)
        print(f"Validation loss for epoch {epoch+1}: {val_loss:.4f}")

        if args.cos_lr:
            lr_scheduler.step()
        else:
            lr_scheduler.step(val_loss)

        # 2) Evaluate on val set
        print("Evaluating on validation set...")
        coco_eval = engine.evaluate(
            model, val_loader, device=device, json_path=json_results_path, epoch=epoch
        )

        if hasattr(coco_eval, "coco_eval") and "bbox" in coco_eval.coco_eval:
            stats = coco_eval.coco_eval["bbox"].stats
            val_mAP_50_95 = stats[0] * 100.0
            val_mAP_50 = stats[1] * 100.0
            if val_mAP_50_95 > best_val_mAP_50_95:
                print(
                    f"[INFO] Validation mAP improved from {best_val_mAP_50_95:.2f} to {val_mAP_50_95:.2f}."
                )
                best_val_mAP_50_95 = val_mAP_50_95
        else:
            val_mAP_50_95 = 0.0
            val_mAP_50 = 0.0

        # 3) Log to CSV
        train_loss = (
            train_stats.meters["loss"].global_avg
            if "loss" in train_stats.meters
            else 0.0
        )
        current_lr = optimizer.param_groups[0]["lr"]

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

        # 4) Save checkpoints
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

        # 5) Early stopping
        early_stopper.update(val_loss)
        if early_stopper.should_stop:
            print("[INFO] Early stopping triggered.")
            break

    # Final checkpoint
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

    # Optional test evaluation
    if test_loader:
        print("\nEvaluating on test set...")
        engine.evaluate(
            model,
            test_loader,
            device=device,
            json_path=os.path.join(results_dir, "test_preds.json"),
            epoch="final",
        )

    print("\nTraining completed!")


###############################################################################
# Argparse Entry
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train object detection models (RetinaNet) with no custom collate"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./scripts/models/args.yaml",
        help="Path to the YAML config file",
    )
    args_cli = parser.parse_args()

    # Load the YAML config and convert to a Namespace
    config_dict = load_config(args_cli.config)
    args = dict_to_namespace(config_dict)

    main(args)
