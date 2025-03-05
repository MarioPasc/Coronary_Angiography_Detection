import os
import shutil
from pathlib import Path
from typing import Union

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


def construct_retinanet(root_folder: Union[str, Path]) -> None:
    """
    Example function (stub) for constructing a RetinaNet-compatible dataset.
    Adjust the logic as needed for your specific format requirements.

    :param root_folder: Path to the root folder containing images/, labels_pascal_voc/, etc.
    """
    root_path = Path(root_folder).resolve()
    retina_path = root_path / "datasets" / "retina_net"
    retina_images = retina_path / "images"
    retina_labels = retina_path / "labels"

    # Ensure the retina_net/ directory structure exists
    retina_images.mkdir(parents=True, exist_ok=True)
    retina_labels.mkdir(parents=True, exist_ok=True)

    # Here you would implement:
    #   - Copying/Linking images
    #   - Copying annotations from Pascal VOC or another source
    #   - Possibly transforming annotation formats
    # For now, it’s just a placeholder
    pass


# Add more functions construct_faster_rcnn, construct_whatever(), etc.
