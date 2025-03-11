import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # type: ignore
import os

# Import the UNet model and dataset
# Assuming the UNet and dataset code is in the same directory
from ICA_Detection.models.segmentation.unet import UNet
from ICA_Detection.splits.holdout.holdout_segmentation_models import (
    holdout_segmentation,
    visualize_sample,
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
json_data_path = "/home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/json_metadata/processed.json"
splits_info_path = (
    "/home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/json_metadata/splits.json"
)
images_root = "/home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/images"
masks_root = (
    "/home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/datasets/segmentation/masks"
)
output_path = "/home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/experiments"
# Hyperparameters
batch_size = 4
num_epochs = 20
learning_rate = 0.0001
img_size = 512


# Define Dice coefficient
def dice_coefficient(pred, target):
    smooth = 1e-5
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )


# Get dataloaders
train_loader, val_loader, test_loader = holdout_segmentation(
    json_data_path=json_data_path,
    splits_info_path=splits_info_path,
    images_root=images_root,
    masks_root=masks_root,
    img_size=img_size,
    batch_size=batch_size,
)

# Initialize model
model = UNet(n_channels=3, n_classes=1).to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
best_val_dice = 0
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    train_dice = 0

    for images, masks in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
    ):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate metrics
        train_loss += loss.item()

        # Convert outputs to binary predictions
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        train_dice += dice_coefficient(predictions, masks).item()

    # Calculate average metrics
    train_loss /= len(train_loader)
    train_dice /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    val_dice = 0

    with torch.no_grad():
        for images, masks in tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
        ):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Calculate metrics
            val_loss += loss.item()

            # Convert outputs to binary predictions
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            val_dice += dice_coefficient(predictions, masks).item()

    # Calculate average metrics
    val_loss /= len(val_loader)
    val_dice /= len(val_loader)

    # Print metrics
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

    # Save best model
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), "best_unet_model.pth")
        print(f"  New best model saved with Dice: {val_dice:.4f}")

    # Visualize predictions (every 5 epochs)
    if (epoch + 1) % 5 == 0:
        # Get a batch of validation data
        val_images, val_masks = next(iter(val_loader))

        # Select one sample
        image = val_images[0:1].to(device)
        mask = val_masks[0:1]

        # Make prediction
        model.eval()
        with torch.no_grad():
            output = model(image)
            pred = torch.sigmoid(output)
            pred_mask = (pred > 0.5).float()

        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Display image
        img_np = val_images[0].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        axes[0].imshow(img_np)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        # Display ground truth mask
        axes[1].imshow(val_masks[0].squeeze().numpy(), cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        # Display predicted mask
        axes[2].imshow(pred_mask[0].squeeze().cpu().numpy(), cmap="gray")
        axes[2].set_title(
            f"Prediction (Dice: {dice_coefficient(pred_mask, mask.to(device)):.4f})"
        )
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"results_epoch_{epoch+1}.png"))
        plt.close()

# Load best model and evaluate on test set
if test_loader is not None:
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load("best_unet_model.pth"))
    model.eval()

    test_dice = 0
    all_dices = []

    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)

            # Convert outputs to binary predictions
            predictions = (torch.sigmoid(outputs) > 0.5).float()

            # Calculate Dice for each sample
            for j in range(images.size(0)):
                sample_dice = dice_coefficient(
                    predictions[j : j + 1], masks[j : j + 1]
                ).item()
                all_dices.append(sample_dice)

                # Visualize a few test predictions
                if i < 5 and j == 0:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    # Display image
                    img_np = images[j].cpu().permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = std * img_np + mean
                    img_np = np.clip(img_np, 0, 1)
                    axes[0].imshow(img_np)
                    axes[0].set_title("Input Image")
                    axes[0].axis("off")

                    # Display ground truth mask
                    axes[1].imshow(masks[j].squeeze().cpu().numpy(), cmap="gray")
                    axes[1].set_title("Ground Truth")
                    axes[1].axis("off")

                    # Display predicted mask
                    axes[2].imshow(predictions[j].squeeze().cpu().numpy(), cmap="gray")
                    axes[2].set_title(f"Prediction (Dice: {sample_dice:.4f})")
                    axes[2].axis("off")

                    plt.tight_layout()
                    plt.savefig(f"test_result_{i}.png")
                    plt.close()

    # Calculate average and per-case statistics
    avg_dice = sum(all_dices) / len(all_dices)
    print(f"Test Set Results:")
    print(f"  Average Dice: {avg_dice:.4f}")
    print(f"  Min Dice: {min(all_dices):.4f}")
    print(f"  Max Dice: {max(all_dices):.4f}")

    # Plot Dice score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_dices, bins=20, alpha=0.7)
    plt.axvline(avg_dice, color="red", linestyle="dashed", linewidth=1)
    plt.text(avg_dice + 0.01, plt.ylim()[1] * 0.9, f"Mean: {avg_dice:.4f}", color="red")
    plt.xlabel("Dice Coefficient")
    plt.ylabel("Number of Samples")
    plt.title("Distribution of Dice Scores on Test Set")
    plt.grid(alpha=0.3)
    plt.savefig("dice_distribution.png")
    plt.close()
