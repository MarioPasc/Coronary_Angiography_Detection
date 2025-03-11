import os
import torch
from typing import Optional, Dict, Any


class EarlyStopping:
    """
    Early stopping mechanism based on a validation loss metric.

    Attributes:
        patience (int): How many epochs to wait without improvement
                        before stopping the training.
        min_delta (float): The minimum absolute change in loss to qualify as an improvement.
        counter (int): Counts how many epochs have passed without improvement.
        best_loss (float): The best (lowest) validation loss seen so far.
        should_stop (bool): Set to True when patience is exceeded.

    Example usage:
        >>> early_stopper = EarlyStopping(patience=3, min_delta=0.0)
        >>> for epoch in range(num_epochs):
        ...     # ... train ...
        ...     val_loss = compute_validation_loss(...)
        ...     early_stopper.update(val_loss)
        ...     if early_stopper.should_stop:
        ...         print("Early stopping triggered")
        ...         break
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0) -> None:
        """
        Args:
            patience (int): The number of epochs to wait without improvement.
            min_delta (float): Minimum change in the monitored value to consider it an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def update(self, current_loss: float) -> None:
        """
        Checks if the current_loss is an improvement over the best_loss.
        If not, increments patience counter. If improvement, reset counter.

        Args:
            current_loss (float): The current epoch's validation loss.
        """
        if (self.best_loss - current_loss) > self.min_delta:
            # We got an improvement
            self.best_loss = current_loss
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def save_best_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_map_50_95: float,
    best_map_50_95_so_far: float,
    checkpoint_dir: str,
    model_name: str = "best_model.pth",
) -> float:
    """
    Saves a checkpoint if the current val_map_50_95 is better than the best so far.

    Args:
        model (torch.nn.Module): The model instance being trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        epoch (int): The current epoch number.
        val_map_50_95 (float): The current epoch's validation mAP (0.5:0.95).
        best_map_50_95_so_far (float): The best val_map_50_95 we've encountered so far.
        checkpoint_dir (str): Directory in which to save the checkpoint.
        model_name (str): File name for the best model checkpoint.

    Returns:
        float: Updated best_map_50_95_so_far (possibly unchanged if no improvement).
    """
    if val_map_50_95 > best_map_50_95_so_far:
        print(
            f"[INFO] val_mAP_50_95 improved from {best_map_50_95_so_far:.2f} to {val_map_50_95:.2f}. "
            f"Saving best checkpoint."
        )
        best_map_50_95_so_far = val_map_50_95

        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_map_50_95": val_map_50_95,
            },
            checkpoint_path,
        )

    return best_map_50_95_so_far


def load_best_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Loads a previously saved "best checkpoint" from disk.

    Args:
        model (torch.nn.Module): The model to restore.
        optimizer (torch.optim.Optimizer): The optimizer to restore (can be None).
        checkpoint_path (str): Path to the .pth file.
        device (torch.device): CPU or CUDA device to map the checkpoint.

    Returns:
        Dict[str, Any]: A dictionary with checkpoint info, including 'epoch' and 'val_map_50_95'.
    """
    print(f"[INFO] Loading best checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
