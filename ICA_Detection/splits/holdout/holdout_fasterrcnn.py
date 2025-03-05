import os
import json
from typing import Optional, Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


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


def _get_patient_key(item_id: str) -> str:
    """
    Given an item's ID (e.g. 'cadica_p26_v1_00020'),
    return the patient group key ('cadica_p26').
    """
    parts = item_id.split("_")
    # Minimal check
    if len(parts) < 2:
        return "unknown_unknown"
    dataset_prefix = parts[0].lower()
    patient_prefix = parts[1].lower()
    return f"{dataset_prefix}_{patient_prefix}"


def holdout_retinanet(
    retinanet_json_path: str,
    splits_info_path: str,
    transforms=None,
    batch_size: int = 4,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Read a full RetinaNet-style JSON (with 'Standard_dataset'),
    and a splits_info.json containing the patient-level splits.
    Then create:
      train_loader, val_loader, and test_loader (optional if non-empty).

    Returns (train_loader, val_loader, test_loader_or_None).
    """
    # 1) Load the entire "master" RetinaNet JSON
    if not os.path.isfile(retinanet_json_path):
        raise FileNotFoundError(f"RetinaNet JSON not found: {retinanet_json_path}")
    with open(retinanet_json_path, "r") as f:
        retinanet_data = json.load(f)

    all_items = list(retinanet_data["Standard_dataset"].values())

    # 2) Load the splits_info.json
    if not os.path.isfile(splits_info_path):
        raise FileNotFoundError(f"Splits info JSON not found: {splits_info_path}")
    with open(splits_info_path, "r") as f:
        splits_info = json.load(f)
    # splits_info has structure like:
    # {
    #   "train": {
    #       "cadica": ["cadica_p26", ...],
    #       "kemerovo": ["kemerovo_p13", ...]
    #   },
    #   "val":   { ... },
    #   "test":  { ... }
    # }

    # 3) Create a quick lookup: patient_key -> which split it belongs to
    #    e.g. "cadica_p26" -> "train"
    patient_to_split = {}
    for split_name, dataset_map in splits_info.items():
        for _, patient_list in dataset_map.items():
            for pkey in patient_list:
                patient_to_split[pkey] = split_name

    # 4) Partition the dataset items into train / val / test
    train_samples = []
    val_samples = []
    test_samples = []

    for item in all_items:
        pid = item["id"]  # e.g. "cadica_p26_v1_00020"
        pkey = _get_patient_key(pid)
        split_name = patient_to_split.get(pkey, None)
        if split_name == "train":
            train_samples.append(item)
        elif split_name == "val":
            val_samples.append(item)
        elif split_name == "test":
            test_samples.append(item)
        else:
            # This item might not be in splits_info at all.
            # You can either ignore or handle as needed.
            pass

    # 5) Create the three Datasets
    train_dataset = FasterRCNNDataset(train_samples, transforms=transforms)
    val_dataset = FasterRCNNDataset(val_samples, transforms=transforms)
    test_dataset = FasterRCNNDataset(test_samples, transforms=transforms)

    # 6) Create DataLoaders
    #    collate_fn must combine multiple samples into a batch
    #    For detection tasks, a common pattern is:
    def collate_fn(batch):
        return list(zip(*batch))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    test_loader = None
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

    return train_loader, val_loader, test_loader
