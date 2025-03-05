import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

import torchvision  # type: ignore
from torchvision.models.detection.retinanet import RetinaNetHead  # type: ignore
from torchvision.transforms import functional as F  # type: ignore

##############################################################################
#  A) Function to create a local dataset folder for RetinaNet
##############################################################################


def create_retinanet_dataset(json_path, output_dir):
    """
    Similar to create_frcnn_dataset but for RetinaNet.
    Symlinks images into output_dir, writes new JSON with updated paths.
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

        if not os.path.exists(symlink_path):
            os.symlink(original_img_path, symlink_path)

        new_item = dict(item)
        new_item["image"] = dict(item["image"])
        new_item["image"]["dataset_route"] = symlink_path
        new_data["Standard_dataset"][key] = new_item

    new_json_path = os.path.join(output_dir, "retinanet_annotations.json")
    with open(new_json_path, "w") as out_f:
        json.dump(new_data, out_f, indent=2)

    return new_json_path


##############################################################################
#  B) Custom Dataset for RetinaNet (no augmentation, label always '1')
##############################################################################


class RetinaNetDataset(Dataset):
    def __init__(self, json_path, transforms=None):
        self.json_path = json_path
        self.transforms = transforms

        with open(json_path, "r") as f:
            data = json.load(f)

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

        for k, v in annots.items():
            if k.startswith("bbox"):
                xmin = float(v["xmin"])
                ymin = float(v["ymin"])
                xmax = float(v["xmax"])
                ymax = float(v["ymax"])
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)  # single lesion class => 1

        # Ensure shape is (N,4) even if N=0
        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)  # shape (N,4)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)

        # Build target
        target = {}
        target["boxes"] = boxes_tensor
        target["labels"] = labels_tensor
        target["image_id"] = torch.tensor([idx])

        if boxes_tensor.shape[0] == 0:
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )
        target["area"] = area

        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        # Optionally apply transforms
        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target


##############################################################################
#  C) Model builder: RetinaNet with ResNet50-FPN
##############################################################################


def get_retinanet_model():
    """
    Returns a RetinaNet model with background + 1 lesion class => num_classes=2.
    """
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)

    # Replace classification head
    in_features = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetHead(
        in_channels=in_features,
        num_anchors=num_anchors,
        num_classes=2,  # background + 1 lesion
        conv_depth=4,
    )
    return model


##############################################################################
#  D) Simple transform
##############################################################################


def basic_transform(image, target):
    return F.to_tensor(image), target
