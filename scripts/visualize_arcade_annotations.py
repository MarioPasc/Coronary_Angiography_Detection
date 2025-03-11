import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon
from PIL import Image
import os
from pathlib import Path
from typing import Union, List, Dict, Any


def visualize_coco_annotations(
    stenosis_coco_path: Union[str, Path],
    arteries_coco_path: Union[str, Path],
    stenosis_image_folder: Union[str, Path],
    arteries_image_folder: Union[str, Path],
    image_id: int,
    output_path: Union[str, Path] = None,
    figsize: tuple = (18, 9),
):
    """
    Visualizes stenosis and arteries annotations from COCO datasets for a given image.

    Args:
        stenosis_coco_path: Path to the stenosis COCO JSON file
        arteries_coco_path: Path to the arteries COCO JSON file
        image_folder: Path to folder containing the images
        image_id: The COCO image ID to visualize
        output_path: Path to save the visualization (if None, will display)
        figsize: Figure size for the plot

    Returns:
        None, but saves or displays the visualization
    """
    # Load the COCO JSON files
    with open(stenosis_coco_path, "r") as f:
        stenosis_data = json.load(f)

    with open(arteries_coco_path, "r") as f:
        arteries_data = json.load(f)

    # Find the image info
    image_info = None
    for img in stenosis_data["images"]:
        if img["id"] == image_id:
            image_info = img
            break

    if image_info is None:
        raise ValueError(
            f"Image with ID {image_id} not found in the stenosis COCO dataset"
        )

    # Load the image
    image_path_stenosis = os.path.join(stenosis_image_folder, image_info["file_name"])
    image_stenosis = np.array(Image.open(image_path_stenosis))

    image_path_arteries = os.path.join(arteries_image_folder, "676.png")
    image_arteries = np.array(Image.open(image_path_arteries))

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"COCO Annotations for Image ID: {image_id}", fontsize=16)

    # Display the image in both subplots
    ax1.imshow(image_stenosis)
    ax1.set_title("Stenosis Annotations")
    ax1.axis("off")

    ax2.imshow(image_arteries)
    ax2.set_title("Arteries Annotations")
    ax2.axis("off")

    # Find and draw stenosis annotations
    stenosis_count = 0
    for ann in stenosis_data["annotations"]:
        if ann["image_id"] == image_id:
            stenosis_count += 1

            # Draw bounding box
            bbox = ann["bbox"]  # [x, y, width, height]
            rect = Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax1.add_patch(rect)

            # Draw segmentation if available
            if "segmentation" in ann:
                seg = ann["segmentation"]
                # Check if seg is a list and not empty
                if isinstance(seg, list) and len(seg) > 0:
                    # If seg is already a flat list, use it directly
                    if all(isinstance(x, (int, float)) for x in seg):
                        flat_seg = seg
                    # Otherwise take the first element if it's a list of lists
                    elif isinstance(seg[0], list):
                        flat_seg = seg[0]
                    else:
                        flat_seg = []

                    len_seg = len(flat_seg)
                    # Convert flat list to [[x1,y1], [x2,y2], ...] format for Polygon
                    polygon_points = []
                    for i in range(0, len_seg, 2):
                        if i + 1 < len_seg:  # Ensure we have both x and y
                            polygon_points.append([flat_seg[i], flat_seg[i + 1]])

                    if polygon_points:
                        polygon = Polygon(
                            polygon_points,
                            closed=False,
                            fill=True,
                            alpha=0.3,
                            color="r",
                        )
                        ax1.add_patch(polygon)

    # Find and draw arteries annotations
    arteries_count = 0
    for ann in arteries_data["annotations"]:
        if ann["image_id"] == image_id:
            arteries_count += 1

            # Draw bounding box
            bbox = ann["bbox"]  # [x, y, width, height]
            rect = Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                linewidth=2,
                edgecolor="b",
                facecolor="none",
            )
            ax2.add_patch(rect)

            # Draw segmentation if available
            if "segmentation" in ann:
                seg = ann["segmentation"]
                # Convert flat list to [[x1,y1], [x2,y2], ...] format for Polygon
                polygon_points = []
                for i in range(0, len(seg), 2):
                    if i + 1 < len(seg):  # Ensure we have both x and y
                        polygon_points.append([seg[i], seg[i + 1]])

                if polygon_points:
                    polygon = Polygon(
                        polygon_points, closed=False, fill=True, alpha=0.3, color="b"
                    )
                    ax2.add_patch(polygon)

    # Add annotation counts to titles
    ax1.set_title(f"Stenosis Annotations: {stenosis_count}")
    ax2.set_title(f"Arteries Annotations: {arteries_count}")

    # Adjust layout
    plt.tight_layout()

    # Save or display the figure
    if output_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, format="svg", bbox_inches="tight")
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)


# Example usage:
def example_visualization():
    """
    Example showing how to use the visualization function.
    """
    # Paths to your COCO JSON files
    stenosis_coco_path = "/home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/datasets/stenosis/stenosis_coco.json"
    arteries_coco_path = "/home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/datasets/arteries/arteries_coco.json"

    # Path to your image folder
    stenosis_image_folder = (
        "/home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/images/images_stenosis"
    )
    arteries_image_folder = (
        "/home/mariopasc/Python/Datasets/COMBINED/ICA_DETECTION/images/images_arteries"
    )
    # Image ID to visualize (this is the COCO image ID, not the original image name)
    image_id = 1  # Change this to the ID you want to visualize

    # Output path for the visualization
    output_path = "./scripts/visualization.svg"

    # Call the visualization function
    visualize_coco_annotations(
        stenosis_coco_path=stenosis_coco_path,
        arteries_coco_path=arteries_coco_path,
        stenosis_image_folder=stenosis_image_folder,
        arteries_image_folder=arteries_image_folder,
        image_id=image_id,
        output_path=output_path,
    )


if __name__ == "__main__":
    example_visualization()
