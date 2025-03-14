import os
import json
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def load_json(json_path: str) -> dict:
    """
    Load JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON content.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_image(image_path: str) -> Image.Image:
    """
    Load an image from the given path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Image.Image: Loaded image.
    """
    return Image.open(image_path)

def plot_bboxes(image: Image.Image, bboxes: List[List[float]], labels: List[int], color: str, ax: plt.Axes) -> None:
    """
    Plot bounding boxes on the image.

    Args:
        image (Image.Image): The image to plot on.
        bboxes (List[List[float]]): List of bounding boxes in [x, y, w, h] format.
        labels (List[int]): List of labels corresponding to the bounding boxes.
        color (str): Color of the bounding boxes.
        ax (plt.Axes): Matplotlib Axes object to plot on.
    """
    for bbox, label in zip(bboxes, labels):
        if bbox[0] == -1:
            continue
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1] - 10, f'Label: {label}', color=color, fontsize=12, weight='bold')

def save_annotated_image(image: Image.Image, save_path: str) -> None:
    """
    Save the annotated image.

    Args:
        image (Image.Image): The image to save.
        save_path (str): Path to save the image.
    """
    image.save(save_path)

def annotate_image(image_folder: str, image_name: str, json_data: dict, save_folder: str) -> None:
    """
    Annotate an image with predicted and ground truth bounding boxes.

    Args:
        image_folder (str): Folder containing the images.
        image_name (str): Name of the image to annotate.
        json_data (dict): JSON data containing bounding box information.
        save_folder (str): Folder to save the annotated images.
    """
    image_path = os.path.join(image_folder, image_name)
    image = load_image(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap="gray")

    epoch_data = json_data.get("epoch_4", {})
    image_data = epoch_data.get(image_name, {})

    predicted_bboxes = image_data.get("predicted_bboxes", [])
    predicted_labels = image_data.get("predicted_labels", [])
    gt_bboxes = image_data.get("gt_bboxes", [])
    gt_labels = image_data.get("gt_labels", [])

    plot_bboxes(image, predicted_bboxes, predicted_labels, 'r', ax)
    plot_bboxes(image, gt_bboxes, gt_labels, 'g', ax)

    save_path = os.path.join(save_folder, f'annotated_{image_name}')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def main(image_folder: str, json_path: str, save_folder: str) -> None:
    """
    Main function to annotate images based on JSON data.

    Args:
        image_folder (str): Folder containing the images.
        json_path (str): Path to the JSON file.
        save_folder (str): Folder to save the annotated images.
    """
    os.makedirs(save_folder, exist_ok=True)
    json_data = load_json(json_path)

    for image_name in json_data.get("epoch_4", {}).keys():
        annotate_image(image_folder, image_name, json_data, save_folder)

if __name__ == "__main__":
    image_folder = "/home/mario/Python/Datasets/COMBINED/tasks/stenosis_detection/images"
    json_path = "/home/mario/x2go_shared/epoch_4_predictions.json"
    save_folder = "/home/mario/Python/Datasets/COMBINED/detection/retina_net"
    main(image_folder, json_path, save_folder)