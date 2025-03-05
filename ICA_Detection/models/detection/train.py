import os
import csv
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  # type: ignore

# Example references to your local modules
from ICA_Detection.models.detection.faster_rcnn import (
    create_frcnn_dataset,
    FasterRCNNDataset,
    get_fasterrcnn_model,
    basic_transform,
)
from ICA_Detection.models.detection.retinanet import (
    create_retinanet_dataset,
    RetinaNetDataset,
    get_retinanet_model,
    basic_transform,
)


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0

    # Create a tqdm progress bar for the training loop
    train_pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Training]", leave=False)

    for images, targets in train_pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass => dict of losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        train_pbar.set_postfix({"loss": f"{losses.item():.4f}"})

    avg_train_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
    print(f"[Train] Epoch {epoch} - Average Loss: {avg_train_loss:.4f}")
    return avg_train_loss


def validate(model, data_loader, device, epoch):
    """
    Returns:
      val_loss, TP, FP, FN, TN, precision, recall, f1
    """
    # -----------------------------
    # (A) Compute Validation Loss
    # -----------------------------
    # TorchVision detection models require train() mode to produce losses,
    # even for validation. We'll disable gradients with torch.no_grad() so we don't update weights.

    model.train()
    total_val_loss = 0.0

    with torch.no_grad():
        val_loss_pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Val Loss]", leave=False)
        for images, targets in val_loss_pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)  # returns a dict of losses
            losses = sum(loss for loss in loss_dict.values())
            total_val_loss += losses.item()

            val_loss_pbar.set_postfix({"val_loss": f"{losses.item():.4f}"})

    avg_val_loss = total_val_loss / len(data_loader) if len(data_loader) > 0 else 0.0

    # -----------------------------
    # (B) Compute Metrics (P, R, F1)
    # -----------------------------
    model.eval()

    TP, FP, FN, TN = 0, 0, 0, 0

    with torch.no_grad():
        val_pred_pbar = tqdm(
            data_loader, desc=f"Epoch {epoch} [Val Metrics]", leave=False
        )
        for images, targets in val_pred_pbar:
            images_gpu = [img.to(device) for img in images]

            # In eval mode + no targets => returns predictions
            preds = model(images_gpu)

            # We'll do a naive image-level approach:
            # If ground truth has >=1 box => GT=1, else GT=0
            # If model predicts >=1 box w/ score>=0.5 => Pred=1, else Pred=0
            for i, pred in enumerate(preds):
                gt_has_box = targets[i]["boxes"].shape[0] > 0

                scores = pred.get("scores", [])
                pred_has_box = (scores >= 0.5).any() if len(scores) else False

                if gt_has_box and pred_has_box:
                    TP += 1
                elif not gt_has_box and pred_has_box:
                    FP += 1
                elif gt_has_box and not pred_has_box:
                    FN += 1
                else:
                    TN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    print(
        f"[Val] Epoch {epoch} - Loss: {avg_val_loss:.4f} | "
        f"P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}, "
        f"TP={TP}, FP={FP}, FN={FN}, TN={TN}"
    )

    return avg_val_loss, TP, FP, FN, TN, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Train + Validate with CSV logging.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["fasterrcnn", "retinanet"],
        help="Which model to train.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to the original metadata JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to create the symlinked dataset + store the final model.",
    )
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument(
        "--csv_log",
        type=str,
        default="training_log.csv",
        help="Path to the CSV file where we store metrics each epoch.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Create the dataset for the chosen model
    if args.model == "fasterrcnn":
        new_json_path = create_frcnn_dataset(args.metadata, args.output_dir)
        dataset = FasterRCNNDataset(new_json_path, transforms=basic_transform)
        model = get_fasterrcnn_model()
    else:
        new_json_path = create_retinanet_dataset(args.metadata, args.output_dir)
        dataset = RetinaNetDataset(new_json_path, transforms=basic_transform)
        model = get_retinanet_model()

    # 2) Train/Val split
    dataset_size = len(dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 3) Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: list(zip(*x)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: list(zip(*x)),
    )

    # 4) Model + device
    model.to(device)

    # 5) Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    # 6) Prepare CSV logging
    csv_path = os.path.join(args.output_dir, args.csv_log)
    # Write header if file doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "TP",
                    "FP",
                    "FN",
                    "TN",
                    "precision",
                    "recall",
                    "f1",
                ]
            )

    # 7) Training loop
    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Validate
        val_loss, TP, FP, FN, TN, precision, recall, f1 = validate(
            model, val_loader, device, epoch
        )

        # 8) Log metrics to CSV
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    val_loss,
                    TP,
                    FP,
                    FN,
                    TN,
                    f"{precision:.4f}",
                    f"{recall:.4f}",
                    f"{f1:.4f}",
                ]
            )

    # 9) Save final model weights
    model_save_path = os.path.join(args.output_dir, f"{args.model}_model_final.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved at: {model_save_path}")


if __name__ == "__main__":
    main()
