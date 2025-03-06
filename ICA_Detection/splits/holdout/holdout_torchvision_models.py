from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
import os
import json
import re
from torch.utils.data import DataLoader
import torch


class SplitCocoDataset(CocoDetection):
    """
    A subclass of CocoDetection that only keeps images from a given 'image_ids' list.
    This way we can filter for train, val, or test sets.
    """

    def __init__(self, root, annFile, image_ids, transforms=None):
        # set transform=None, target_transform=None so it doesn't conflict
        super().__init__(root, annFile, transform=None, target_transform=None)
        self.transforms = transforms  # your detection transform
        self.image_ids_set = set(image_ids)

        # keep_indices logic
        self.keep_indices = []
        for i, img_info in enumerate(self.coco.dataset["images"]):
            if img_info["id"] in self.image_ids_set:
                self.keep_indices.append(i)

    def __len__(self):
        return len(self.keep_indices)

    def __getitem__(self, idx):
        real_idx = self.keep_indices[idx]
        img, anns = super().__getitem__(real_idx)  # anns is a list of dicts

        # Convert annotations to PyTorch-friendly format
        target = self.convert_coco_annotations(anns, real_idx)

        return img, target

    def convert_coco_annotations(self, anns, real_idx):
        # Retrieve the actual image info from self.coco.dataset["images"]
        image_info = self.coco.dataset["images"][real_idx]
        actual_image_id = image_info["id"]

        target = {
            "boxes": [],
            "labels": [],
            "image_id": torch.tensor([actual_image_id], dtype=torch.int64),
            "area": [],
            "iscrowd": [],
        }

        for ann in anns:
            xmin, ymin, width, height = ann["bbox"]
            xmax = xmin + width
            ymax = ymin + height
            target["boxes"].append(
                [xmin, ymin, xmax, ymax]
            )  # Convert to [x1, y1, x2, y2]
            target["labels"].append(ann["category_id"])
            target["area"].append(ann["area"])
            target["iscrowd"].append(ann["iscrowd"])

        # Convert lists to tensors
        target["boxes"] = (
            torch.tensor(target["boxes"], dtype=torch.float32)
            if target["boxes"]
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        target["labels"] = (
            torch.tensor(target["labels"], dtype=torch.int64)
            if target["labels"]
            else torch.zeros((0,), dtype=torch.int64)
        )
        target["area"] = (
            torch.tensor(target["area"], dtype=torch.float32)
            if target["area"]
            else torch.zeros((0,), dtype=torch.float32)
        )
        target["iscrowd"] = (
            torch.tensor(target["iscrowd"], dtype=torch.int64)
            if target["iscrowd"]
            else torch.zeros((0,), dtype=torch.int64)
        )

        return target


def holdout_coco(
    coco_json_path: str,
    splits_info_path: str,
    images_root: str,
    transforms=None,
    batch_size: int = 4,
    shuffle_train: bool = True,
):
    """
    Load a single COCO dataset, read the splits_info.json,
    separate it into train/val/test subsets by 'patient key' or some ID,
    and return (train_loader, val_loader, test_loader).
    """
    # 1) Read splits info
    if not os.path.isfile(splits_info_path):
        raise FileNotFoundError(f"Splits info JSON not found: {splits_info_path}")

    with open(splits_info_path, "r") as f:
        splits_info = json.load(f)

    # Build a dict: "patient_key" -> "train"/"val"/"test"
    patient_to_split = {}
    for split_name, dataset_map in splits_info.items():
        for _, patient_list in dataset_map.items():
            for pkey in patient_list:
                patient_to_split[pkey] = split_name

    # 2) Load the entire COCO
    #    We'll read the annotation file and the root folder with images
    if not os.path.isfile(coco_json_path):
        raise FileNotFoundError(f"COCO JSON not found: {coco_json_path}")

    # We'll parse the COCO JSON ourselves to find image->patient_key
    coco_gt = COCO(coco_json_path)
    all_image_ids = coco_gt.getImgIds()

    # 3) Build train_ids, val_ids, test_ids
    train_ids = []
    val_ids = []
    test_ids = []

    for img_id in all_image_ids:
        # get image info
        info = coco_gt.loadImgs([img_id])[0]
        filename = info["file_name"]  # e.g. "cadica_p26_v1_00020.jpg"
        # parse the patient key from the filename
        # you might do it via regex or your own logic
        # let's suppose it's always "cadica_p26" as prefix:
        # e.g. "cadica_p26_v1_00020.jpg" => "cadica_p26"
        # a quick approach:
        match = re.match(r"([a-zA-Z0-9]+_[a-zA-Z0-9]+)_.*", filename)
        if match:
            pkey = match.group(1)  # e.g. "cadica_p26"
        else:
            # fallback if we can't parse
            pkey = "unknown_unknown"

        # which split does it belong to?
        split_name = patient_to_split.get(pkey, None)
        if split_name == "train":
            train_ids.append(img_id)
        elif split_name == "val":
            val_ids.append(img_id)
        elif split_name == "test":
            test_ids.append(img_id)
        else:
            # not in splits => ignore or handle
            pass

    # 4) Build three subset datasets using SplitCocoDataset
    train_dataset = SplitCocoDataset(
        root=images_root,
        annFile=coco_json_path,
        image_ids=train_ids,
        transforms=transforms,
    )
    val_dataset = SplitCocoDataset(
        root=images_root,
        annFile=coco_json_path,
        image_ids=val_ids,
        transforms=transforms,
    )
    test_dataset = SplitCocoDataset(
        root=images_root,
        annFile=coco_json_path,
        image_ids=test_ids,
        transforms=transforms,
    )

    # 5) Build DataLoaders
    def collate_fn(batch):
        # CocoDetection returns [(img, target), (img, target), ...]
        # We want: [imgs], [targets]
        imgs = []
        targets = []
        for img, tgt in batch:
            imgs.append(img)
            targets.append(tgt)
        return imgs, targets

    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=collate_fn,
    )
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = None
    if len(test_ids) > 0:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

    return train_loader, val_loader, test_loader
