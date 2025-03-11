import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import os
from PIL import Image
import cv2
from typing import List, Dict, Any, Optional

plt.style.use("dark_background")


def visualize_image_with_annotations(json_data, image_id=None, ax=None):
    """
    Visualize an image with its bounding boxes, segmentations, and vessel segmentations.

    Args:
        json_data (dict): Dictionary containing the dataset
        image_id (str, optional): The ID of the image to visualize. If None, the first image with lesions is chosen.
        ax (matplotlib.axes, optional): The axis to draw on. If None, a new one is created.

    Returns:
        tuple: (fig, ax) - The figure and axis objects, or None if visualization failed
    """
    # If no image_id is specified, use the first one with lesions
    if image_id is None:
        for key, entry in json_data.items():
            if entry.get("lesion", False):
                image_id = key
                break

    if image_id not in json_data:
        print(f"Image ID '{image_id}' not found in the dataset.")
        return None, None

    entry = json_data[image_id]
    image_info = entry.get("image", {})
    image_path = (
        image_info.get("route")
        or image_info.get("dataset_route")
        or image_info.get("original_route")
    )

    if not image_path or not os.path.exists(image_path):
        print(f"Image file not found at {image_path}")
        return None, None

    # Load the image
    image = np.array(Image.open(image_path))

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    ax.imshow(image, cmap="gray")

    # Process annotations
    annotations = entry.get("annotations", {})
    colors = plt.cm.tab10.colors  # Use a colormap for different colors

    # Group bbox and segmentation by index
    for i in range(1, 100):  # Assuming there won't be more than 100 annotations
        bbox_key = f"bbox{i}"
        seg_key = f"segmentation{i}"

        if bbox_key not in annotations:
            break

        # Get bounding box
        bbox = annotations[bbox_key]
        xmin = bbox["xmin"]
        ymin = bbox["ymin"]
        xmax = bbox["xmax"]
        ymax = bbox["ymax"]
        label = bbox.get("label", "unknown")

        # Draw bounding box
        color = colors[(i - 1) % len(colors)]
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        # Add label to bounding box
        ax.text(
            xmin,
            ymin - 5,
            f"{i}: {label}",
            color=color,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7),
        )

        # Draw segmentation if available
        if seg_key in annotations:
            seg = annotations[seg_key]
            xyxy = seg.get("xyxy", [])

            # Convert flat list to points array
            if xyxy and len(xyxy) >= 4:
                points = np.array(xyxy).reshape(-1, 2)
                # Draw a polygon
                polygon = patches.Polygon(
                    points,
                    closed=True,
                    fill=True,
                    alpha=0.3,
                    edgecolor=color,
                    facecolor=color,
                )
                ax.add_patch(polygon)

    # Draw vessel segmentations if available
    vessel_segmentations = annotations.get("vessel_segmentations", [])
    vessel_colors = plt.cm.Paired.colors  # Different colormap for vessels

    for i, vessel_seg in enumerate(vessel_segmentations):
        xyxy = vessel_seg.get("xyxy", [])
        label = vessel_seg.get("label", "vessel")

        # Convert flat list to points array
        if xyxy and len(xyxy) >= 4:
            points = np.array(xyxy).reshape(-1, 2)
            # Draw a polygon with different style
            color = vessel_colors[i % len(vessel_colors)]
            polygon = patches.Polygon(
                points,
                closed=True,
                fill=True,
                alpha=0.7,
                edgecolor=color,
                facecolor=color,
                linestyle="--",
                linewidth=1.5,
            )
            ax.add_patch(polygon)

            # Add label for vessel segmentation (at center of polygon)
            if len(points) > 0:
                center_x = np.mean(points[:, 0])
                center_y = np.mean(points[:, 1])
                ax.text(
                    center_x,
                    center_y,
                    f"V{i+1}: {label}",
                    color=color,
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.5),
                    ha="center",
                    va="center",
                )

    # Set title and remove ticks for cleaner visualization
    ax.set_title(f"Image: {image_id}", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def visualize_dataset_samples(
    json_path, num_samples=5, figsize=(15, 5), random_seed=None
):
    """
    Visualize multiple random samples from the dataset in a 1xN subplot.

    Args:
        json_path (str): Path to the JSON file containing the dataset
        num_samples (int, optional): Number of samples to visualize. Default is 5.
        figsize (tuple, optional): Figure size. Default is (15, 5).
        random_seed (int, optional): Seed for random selection. Default is None.

    Returns:
        None: Displays the plots
    """
    # Load data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Get the actual dataset (handle both formats)
    if "Standard_dataset" in data:
        data = data["Standard_dataset"]

    # Get all image IDs with lesions
    image_ids = [key for key, entry in data.items() if entry.get("lesion", False)]

    if len(image_ids) == 0:
        print("No images with lesions found in the dataset.")
        return

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Select random images
    if len(image_ids) <= num_samples:
        selected_ids = image_ids
        num_samples = len(image_ids)
    else:
        selected_ids = np.random.choice(image_ids, size=num_samples, replace=False)

    # Create a 1xN subplot figure
    fig, axes = plt.subplots(
        1, num_samples, figsize=(figsize[0] * num_samples // 5, figsize[1])
    )

    # Ensure axes is always an array even with one sample
    if num_samples == 1:
        axes = [axes]

    # Visualize each selected image
    for i, image_id in enumerate(selected_ids):
        _, _ = visualize_image_with_annotations(data, image_id, ax=axes[i])

    plt.tight_layout()
    plt.show()
    return fig


def save_visualization(
    json_path, output_path, num_samples=5, figsize=(15, 5), random_seed=None
):
    """
    Visualize and save multiple random samples from the dataset to a file.

    Args:
        json_path (str): Path to the JSON file containing the dataset
        output_path (str): Path to save the visualization
        num_samples (int, optional): Number of samples to visualize. Default is 5.
        figsize (tuple, optional): Figure size. Default is (15, 5).
        random_seed (int, optional): Seed for random selection. Default is None.

    Returns:
        None: Saves the visualization to a file
    """
    fig = visualize_dataset_samples(json_path, num_samples, figsize, random_seed)
    if fig:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    visualize_dataset_samples(
        json_path="/home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/json_metadata/processed.json",
        num_samples=4,
        figsize=(20, 5),
    )

    save_visualization(
        json_path="/home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/json_metadata/processed.json",
        output_path="visualization.svg",
        num_samples=4,
        figsize=(20, 5),
    )
