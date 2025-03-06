import os
import shutil
from pathlib import Path
from typing import Union, Any, Dict
import json


def construct_yolo(root_folder: Union[str, Path]) -> None:
    """
    Create the YOLO dataset folder structure under root_folder/datasets/yolo
    by:
      1. Creating the subfolders: images/, labels/
      2. Making symbolic links in images/ to all files under root_folder/images
      3. Copying (or renaming) labels_yolo/ to labels/

    :param root_folder: Path to the root folder containing images/, labels_yolo/, etc.
    """
    root_path = Path(root_folder).resolve()
    yolo_path = root_path / "datasets" / "yolo"
    yolo_images = yolo_path / "images"
    yolo_labels = yolo_path / "labels"

    # Ensure the yolo/ directory structure exists
    yolo_images.mkdir(parents=True, exist_ok=True)
    yolo_labels.mkdir(parents=True, exist_ok=True)

    # 1) Create symbolic links for images
    original_images_dir = root_path / "images"
    for image_file in original_images_dir.iterdir():
        if image_file.is_file():
            # e.g., "cadica_p26_v1_00020.png"
            link_dest = yolo_images / image_file.name
            # os.symlink(src=..., dst=...) must not exist at dst
            if link_dest.exists():
                link_dest.unlink()
            os.symlink(image_file, link_dest)

    # 2) Copy YOLO labels
    original_labels_yolo = root_path / "labels_yolo"
    if original_labels_yolo.exists() and original_labels_yolo.is_dir():
        for label_file in original_labels_yolo.iterdir():
            if label_file.is_file():
                shutil.copy(label_file, yolo_labels / label_file.name)


def construct_pytorch_compatible(
    json_path: Union[str, Path], root_folder: Union[str, Path], dataset_name: str
) -> str:
    """
    Creates a RetinaNet-compatible dataset folder. It symlinks images into
    output_dir/images, and writes a new JSON file with updated image paths.

    :param json_path: Path to the original processed.json
    :param output_dir: Folder where the new dataset structure will be created
    :return: Path to the newly created JSON annotation file
    """
    root_path = Path(root_folder).resolve()
    datasets_path = root_path / "datasets"
    datasets_path.mkdir(parents=True, exist_ok=True)

    dataset_name = dataset_name.strip("_")

    output_dir = datasets_path / dataset_name
    os.makedirs(output_dir, exist_ok=True)

    # Avoid creating unnecessary symlinks
    """
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    new_data: Dict[str, Any] = {"Standard_dataset": {}}

    for key, item in data["Standard_dataset"].items():
        original_img_path = item["image"]["dataset_route"]

        # Avoid creating unnecessary symlinks
        """
        filename = os.path.basename(original_img_path)
        symlink_path = os.path.join(images_dir, filename)

        if not os.path.exists(symlink_path):
            os.symlink(original_img_path, symlink_path)
        """
        # Copy item data, updating the "dataset_route" to the new symlink path
        new_item = dict(item)
        new_item["image"] = dict(item["image"])
        new_item["image"]["dataset_route"] = original_img_path

        new_data["Standard_dataset"][key] = new_item

    new_json_path = os.path.join(output_dir, f"{dataset_name}_annotations.json")
    with open(new_json_path, "w") as out_f:
        json.dump(new_data, out_f, indent=2)

    return new_json_path
