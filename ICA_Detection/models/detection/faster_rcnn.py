import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

import torchvision  # type: ignore
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # type: ignore
from torchvision.transforms import functional as F  # type: ignore

##############################################################################
#  A) Function to create a local dataset folder for Faster R-CNN
##############################################################################


def create_frcnn_dataset(json_path, output_dir):
    """
    Reads the metadata JSON, creates a new dataset folder for Faster R-CNN,
    symlinks images, and writes a new JSON with updated `dataset_route`.

    Args:
        json_path (str): Path to original metadata JSON.
        output_dir (str): Destination folder for this model's dataset.

    Returns:
        str: Path to the new JSON file that references the symlinked images.
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    new_data = {"Standard_dataset": {}}

    for key, item in data["Standard_dataset"].items():
        original_img_path = item["image"]["dataset_route"]
        filename = os.path.basename(original_img_path)
        symlink_path = os.path.join(images_dir, filename)

        # Create a symlink to the original image (if it doesn't exist yet)
        if not os.path.exists(symlink_path):
            os.symlink(original_img_path, symlink_path)

        # Copy the item so we can modify the route
        new_item = dict(item)
        new_item["image"] = dict(item["image"])
        new_item["image"]["dataset_route"] = symlink_path
        new_data["Standard_dataset"][key] = new_item

    # Write new JSON
    new_json_path = os.path.join(output_dir, "faster_rcnn_annotations.json")
    with open(new_json_path, "w") as out_f:
        json.dump(new_data, out_f, indent=2)

    return new_json_path


##############################################################################
#  B) Custom Dataset for Faster R-CNN (no augmentation, label always '1')
##############################################################################


class FasterRCNNDataset(Dataset):
    def __init__(self, json_path, transforms=None):
        self.json_path = json_path
        self.transforms = transforms

        with open(json_path, "r") as f:
            data = json.load(f)

        # List of all samples
        self.samples = list(data["Standard_dataset"].values())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Load image
        img_path = item["image"]["dataset_route"]
        img = Image.open(img_path).convert("RGB")

        # Parse bounding boxes
        annots = item.get("annotations", {})
        boxes = []
        labels = []

        # Loop over all possible bboxes: bbox1, bbox2, ...
        for k, v in annots.items():
            if k.startswith("bbox"):
                xmin = float(v["xmin"])
                ymin = float(v["ymin"])
                xmax = float(v["xmax"])
                ymax = float(v["ymax"])
                # Here we assume a single lesion class => label=1
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)

        # Ensure shape is (N,4) even if N=0
        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)  # shape (N,4)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes_tensor
        target["labels"] = labels_tensor
        target["image_id"] = torch.tensor([idx])

        # If there are no boxes, this shape is (0,)
        if boxes_tensor.shape[0] == 0:
            # Negative example: no bounding boxes
            area = torch.tensor([], dtype=torch.float32)
        else:
            # area has shape (N,)
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )

        target["area"] = area
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        # Apply transforms (if any)
        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target


##############################################################################
#  C) Model builder: Faster R-CNN with ResNet50-FPN
##############################################################################


def get_fasterrcnn_model():
    """
    Returns a Faster R-CNN model with background + 1 lesion class => num_classes = 2.
    """
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the box predictor to match num_classes=2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    return model


##############################################################################
#  D) Simple transform
##############################################################################


def basic_transform(image, target):
    # No augmentation: just convert PIL to tensor
    return F.to_tensor(image), target
