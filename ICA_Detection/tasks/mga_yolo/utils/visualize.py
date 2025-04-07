from typing import Dict, Any
import os
import matplotlib.pyplot as plt
from PIL import Image
import logging


def create_mga_visualization(
    visualization_data: Dict[str, Any], batch_idx: int, output_dir: str
) -> str:
    """
    Create and save a visualization of the Mask-Guided Attention process.

    Args:
        visualization_data: Dictionary containing feature maps and masks
        batch_idx: Current batch index for naming
        output_dir: Directory to save visualizations

    Returns:
        Path to the saved visualization
    """
    # Check for required data
    if visualization_data["original_input"] is None:
        logging.error("Cannot visualize: original_input is None")
        return ""

    # Create visualization directory
    os.makedirs(output_dir, exist_ok=True)

    # Create figure
    plt.figure(figsize=(15, 20))

    # Plot original image
    plt.subplot(4, 3, 1)
    try:
        input_img = Image.open(visualization_data["original_input"])
        plt.imshow(input_img)
        plt.title("Input Image")
    except Exception as e:
        logging.error(f"Error opening image: {e}")
        plt.text(0.5, 0.5, "Image load error", ha="center", va="center")
    plt.axis("off")

    # Plot mask
    plt.subplot(4, 3, 2)
    plt.imshow(visualization_data["mask"], cmap="gray")
    plt.title("Segmentation Mask")
    plt.axis("off")

    # Empty predictions slot
    plt.subplot(4, 3, 3)
    plt.title("Predictions (not implemented)")
    plt.axis("off")

    # Feature map layer names to display
    layers = ["model.15", "model.18", "model.21"]
    layer_titles = ["P3", "P4", "P5"]

    # Plot feature maps, masks, and masked feature maps
    _plot_feature_maps(visualization_data, layers, layer_titles)

    # Save and return path
    save_path = os.path.join(output_dir, f"mga_visualization_batch_{batch_idx}.png")
    plt.savefig(save_path)
    plt.close()

    return save_path


def _plot_feature_maps(
    visualization_data: Dict[str, Any], layers: list, layer_titles: list
) -> None:
    """Helper function to plot feature maps in visualization grid."""
    # Original feature maps
    for i, (layer, title) in enumerate(zip(layers, layer_titles)):
        plt.subplot(4, 3, 4 + i)
        if layer in visualization_data["feature_maps"]:
            feature_map = visualization_data["feature_maps"][layer][0].mean(0).cpu()
            plt.imshow(feature_map.numpy(), cmap="viridis")
            plt.title(f"Feature Map {title}")
            plt.axis("off")
        else:
            plt.title(f"No Feature Map for {title}")

    # Downsized masks
    for i, (layer, title) in enumerate(zip(layers, layer_titles)):
        plt.subplot(4, 3, 7 + i)
        if layer in visualization_data["downsized_masks"]:
            mask = visualization_data["downsized_masks"][layer][0].cpu()
            plt.imshow(mask.numpy(), cmap="gray")
            plt.title(f"Mask for {title}")
            plt.axis("off")
        else:
            plt.title(f"No Mask for {title}")

    # Masked feature maps
    for i, (layer, title) in enumerate(zip(layers, layer_titles)):
        plt.subplot(4, 3, 10 + i)
        if layer in visualization_data["masked_feature_maps"]:
            feature_map = (
                visualization_data["masked_feature_maps"][layer][0].mean(0).cpu()
            )
            plt.imshow(feature_map.numpy(), cmap="viridis")
            plt.title(f"Masked FM {title}")
            plt.axis("off")
        else:
            plt.title(f"No Masked FM for {title}")
