import os
import re
import json
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

from PIL import Image

# For debugging (optional)
DEBUG: bool = False


class RetinaSplitCocoDataset(Dataset):
    """
    A COCO-style dataset compatible with the RetinaNet training code.
    It:
      - Loads images/annotations from the given annFile (COCO JSON)
      - Uses only the subset of images whose IDs are in `image_ids`
      - Returns data in the format: {'img': <H x W x 3>, 'annot': <N x 5>}
        with the last column of 'annot' being the class label.
      - Optionally applies a transform on the sample (dictionary).
    """

    def __init__(self, root, annFile, image_ids, transform=None):
        super().__init__()

        self.root = root
        self.annFile = annFile
        self.transform = transform

        self.coco = COCO(self.annFile)

        # Filter the images by the subset of image_ids we want
        all_img_ids = self.coco.getImgIds()
        image_ids_set = set(image_ids)
        # Keep only those in the intersection
        self.image_ids = [img_id for img_id in all_img_ids if img_id in image_ids_set]

        # Preload category info
        self.load_classes()

    def load_classes(self):
        """
        Load category labels from COCO, and build:
          - self.classes: {class_name -> class_index}
          - self.labels: {class_index -> class_name}
          - self.coco_labels: {class_index -> coco_category_id}
          - self.coco_labels_inverse: {coco_category_id -> class_index}
        """
        categories = self.coco.loadCats(self.coco.getCatIds())
        # Sort by COCO `id` to ensure consistent ordering
        categories.sort(key=lambda x: x["id"])

        self.classes = {}
        self.labels = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}

        for idx, cat in enumerate(categories):
            class_id = idx
            coco_id = cat["id"]
            name = cat["name"]

            # Map internal idx <-> coco_id, and name -> idx
            self.classes[name] = class_id
            self.labels[class_id] = name
            self.coco_labels[class_id] = coco_id
            self.coco_labels_inverse[coco_id] = class_id

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            # We are going to be using torchvision/references/detection 
            # transforms, therefore we must make a distintion between images
            # and labels when passing
            sample = self.transform(img, annot)
            # Now, convert to the RetinaNet dictionary inputs
            sample = {'img': sample[0], 'annot': sample[1]}

        return sample

    def load_image(self, image_id):
        """
        Load the image by ID using PIL and return it as a NumPy array in float32 [0,1].
        """
        image_info = self.coco.loadImgs(image_id)[0]

        full_img_path = os.path.join(self.root, image_info["file_name"])

        # Open image with PIL
        img = Image.open(full_img_path).convert("RGB")  # Ensure 3 channels

        return img

    def load_annotations(self, image_id):
        """
        Returns an Nx5 array of [x1, y1, x2, y2, class_label].
        If there are no annotations, returns an empty (0,5) array.
        """
        ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
        if len(ann_ids) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        coco_anns = self.coco.loadAnns(ann_ids)

        annotations = []

        for ann in coco_anns:
            # Skip degenerate bounding boxes
            x, y, w, h = ann["bbox"]
            if w < 1 or h < 1:
                continue

            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            # Convert category_id to internal label index
            cls_id = self.coco_label_to_label(ann["category_id"])

            annotations.append([x1, y1, x2, y2, cls_id])

        if len(annotations) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        return np.array(annotations, dtype=np.float32)

    def coco_label_to_label(self, coco_label):
        """
        Map from COCO's category_id to our 0-based index.
        """
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        """
        Map from our 0-based index to the original COCO category_id.
        """
        return self.coco_labels[label]

    def image_aspect_ratio(self, idx):
        """
        Ratio = width / height for the image at `idx`.
        Needed by the RetinaNet sampler to group images by aspect ratio.
        """
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        return float(image_info["width"]) / float(image_info["height"])

    def num_classes(self):
        """
        Number of classes in COCO. Default is 80, but if you have a smaller subset,
        you can adapt accordingly or just keep 80 if using the standard categories.
        """
        return len(self.classes)


def collater(data):
    imgs = [s["img"] for s in data]
    annots = [s["annot"] for s in data]
    scales = [s.get('scale', 1.0) for s in data]

    widths = [int(s.shape[2]) for s in imgs]  # Corrected to get width from the third dimension
    heights = [int(s.shape[1]) for s in imgs]  # Corrected to get height from the second dimension
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, 3, max_height, max_width)

    if DEBUG:
        print(f"[DEBUG] Batch size: {batch_size}")
        print(f"[DEBUG] Max width: {max_width}, Max height: {max_height}")
        print(f"[DEBUG] Padded images shape: {padded_imgs.shape}")

    for i in range(batch_size):
        img = imgs[i]
        if DEBUG: print(f"[DEBUG] Image {i} shape: {img.shape}")
        padded_imgs[i, :, : int(img.shape[1]), : int(img.shape[2])] = img  # Corrected to use height and width from the correct dimensions

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, : annot.shape[0], :] = torch.tensor(annot)
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    if DEBUG: print(f"[DEBUG] Annotation padded shape: {annot_padded.shape}")

    return {"img": padded_imgs, "annot": annot_padded, "scale": scales}


def holdout_coco_retina(
    coco_json_path: str,
    splits_info_path: str,
    images_root: str,
    transform=None,
    batch_size: int = 4,
    shuffle_train: bool = True,
):
    """
    Like your original holdout_coco, but instantiates
    `RetinaSplitCocoDataset` instead of `SplitCocoDataset`.
    """
    if not os.path.isfile(splits_info_path):
        raise FileNotFoundError(f"Splits info JSON not found: {splits_info_path}")

    if not os.path.isfile(coco_json_path):
        raise FileNotFoundError(f"COCO JSON not found: {coco_json_path}")

    # 1) Read splits info
    with open(splits_info_path, "r") as f:
        splits_info = json.load(f)

    # Build a dict: patient_key -> "train"/"val"/"test"
    patient_to_split = {}
    for split_name, dataset_map in splits_info.items():
        for _, patient_list in dataset_map.items():
            for pkey in patient_list:
                patient_to_split[pkey] = split_name

    # 2) Use pycocotools to get all COCO images
    coco_gt = COCO(coco_json_path)
    all_image_ids = coco_gt.getImgIds()

    train_ids = []
    val_ids = []
    test_ids = []

    for img_id in all_image_ids:
        info = coco_gt.loadImgs([img_id])[0]
        filename = info["file_name"]
        # Extract the patient key (example with regex).
        match = re.match(r"([a-zA-Z0-9]+_[a-zA-Z0-9]+)_.*", filename)
        if match:
            pkey = match.group(1)
        else:
            pkey = "unknown_unknown"

        split_name = patient_to_split.get(pkey, None)
        if split_name == "train":
            train_ids.append(img_id)
        elif split_name == "val":
            val_ids.append(img_id)
        elif split_name == "test":
            test_ids.append(img_id)
        else:
            pass  # or consider putting it somewhere default

    # 3) Instantiate the three subsets
    train_dataset = RetinaSplitCocoDataset(
        root=images_root,
        annFile=coco_json_path,
        image_ids=train_ids,
        transform=transform,
    )
    val_dataset = RetinaSplitCocoDataset(
        root=images_root,
        annFile=coco_json_path,
        image_ids=val_ids,
        transform=transform,
    )
    test_dataset = RetinaSplitCocoDataset(
        root=images_root,
        annFile=coco_json_path,
        image_ids=test_ids,
        transform=transform,
    )

    # 4) Build DataLoaders

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=collater,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collater
    )
    test_loader = None
    if len(test_ids) > 0:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collater
        )

    if DEBUG:
        print(
            f"[DEBUG] #Train: {len(train_ids)}, #Val: {len(val_ids)}, #Test: {len(test_ids)}"
        )
        # Show first sample
        sample = train_dataset[0]
        print(
            "[DEBUG] Example sample keys:",
            sample.keys(),
            "img shape:",
            sample["img"].shape,
        )

    return train_loader, val_loader, test_loader
