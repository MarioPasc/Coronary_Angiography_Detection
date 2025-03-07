import os
import csv
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# from references/detection code
from ICA_Detection.external.torchvision.detection import (
    engine,
    utils,
    transforms as rtransforms,  # the reference/detection/transforms.py
    presets,  # optional, if you want the provided 'presets.py' transforms
)


# Model definitions
from ICA_Detection.models.detection.faster_rcnn import get_faster_rcnn_model
from ICA_Detection.models.detection.retinanet import get_retina_net_model
from ICA_Detection.models.detection.ssd import get_ssd_model

# Data loader function
from ICA_Detection.splits.holdout.holdout_torchvision_models import holdout_coco

###############################################################################
# 1) Example basic transform (if you only do image transforms, no box transforms)
###############################################################################


def get_transform():
    # If you only need "ToTensor", you can do:
    return rtransforms.Compose(
        [
            rtransforms.PILToTensor(),
            rtransforms.ToDtype(dtype=torch.float32, scale=True),
        ]
    )


###############################################################################
# 2) Main training script
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

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

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

    ############################################################################
    # Optimizer and LR Scheduler
    ############################################################################
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    lr_scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

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

        # --------------------------
        # a) Train one epoch
        # --------------------------
        # engine.train_one_epoch returns a dict of average losses
        train_stats = engine.train_one_epoch(
            model, optimizer, train_loader, device, epoch, print_freq=args.print_freq
        )
        # This call includes the standard warmup and logging logic if you want it.

        # Step the LR scheduler
        lr_scheduler.step()

        # --------------------------
        # b) Evaluate on val set
        # --------------------------
        # engine.evaluate returns a CocoEvaluator if it's recognized as COCO
        print("Evaluating on validation set...")
        coco_eval = engine.evaluate(model, val_loader, device=device)

        # The main AP results are typically in coco_eval.coco_eval["bbox"].stats
        # if your dataset is recognized as COCO.
        # stats[0] = mAP@0.5:0.95, stats[1] = mAP@0.5, ...
        if hasattr(coco_eval, "coco_eval") and "bbox" in coco_eval.coco_eval:
            stats = coco_eval.coco_eval["bbox"].stats
            val_mAP_50_95 = stats[0] * 100.0
            val_mAP_50 = stats[1] * 100.0
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

        # --------------------------
        # c) Save checkpoint
        # --------------------------
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

    # -----------------------------
    # Final test if needed
    # -----------------------------
    if test_loader:
        print("\nEvaluating on test set...")
        engine.evaluate(model, test_loader, device=device)
        # If recognized as COCO, this prints final test AP

    print("\nTraining completed!")


###############################################################################
# 4) Argparse
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train object detection models using references/detection scripts"
    )

    # Dataset parameters
    parser.add_argument(
        "--json_path", type=str, required=True, help="Path to the COCO JSON file"
    )
    parser.add_argument(
        "--splits_info_path",
        type=str,
        required=True,
        help="Path to the splits info JSON file",
    )
    parser.add_argument(
        "--images_root",
        type=str,
        required=True,
        help="Folder with all images (file_name in coco json is relative to this)",
    )

    # Model parameters
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["faster_rcnn", "retina_net", "ssd"],
        help="Which detection model to train?",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained backbone from torchvision",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes (including background)",
    )
    parser.add_argument(
        "--freeze_until",
        type=int,
        default=0,
        help="How many layers to freeze in the backbone",
    )

    # Training hyperparams
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train for"
    )
    parser.add_argument("--lr", type=float, default=0.005, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0005, help="Weight decay"
    )
    parser.add_argument(
        "--lr_step_size", type=int, default=8, help="LR scheduler step size"
    )
    parser.add_argument(
        "--lr_gamma", type=float, default=0.1, help="LR scheduler gamma"
    )

    # Logging & checkpoint
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="Base directory to save outputs",
    )
    parser.add_argument(
        "--resume", type=str, default="", help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=20,
        help="Print frequency during training loop",
    )
    parser.add_argument(
        "--save_freq", type=int, default=5, help="Checkpoint save frequency (epochs)"
    )

    args = parser.parse_args()
    main(args)

"""
python scripts/models/train_eval_torchvision.py --json_path /home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/datasets/retinanet/COCO_coco.json --images_root /home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/images --output_dir /home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/experiments --splits_info_path /home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/json_metadata/splits.json --model_type retina_net --pretrained --freeze_until 2 --epochs 5 --lr 0.0001 --lr_step_size 2 --lr_gamma 0.1 --print_freq 1 --save_freq 1 
"""
