import os
import shutil
import glob
import argparse
from pathlib import Path


def transform_dataset(detection_path, masks_path, output_path, yaml_path=None):
    """
    Transform dataset by finding intersection between detection dataset and masks,
    maintaining the train/val/test split structure.

    Args:
        detection_path: Path to YOLO detection dataset
        masks_path: Path to segmentation masks folder
        output_path: Path to output directory
        yaml_path: Path to YAML file (optional)
    """
    # Create output directories
    output_detection_path = os.path.join(output_path, "YOLO_MGA")
    output_detection_dir = os.path.join(output_detection_path, "detection")
    output_images_dir = os.path.join(output_detection_dir, "images")
    output_labels_dir = os.path.join(output_detection_dir, "labels")
    output_masks_dir = os.path.join(output_detection_path, "masks")

    # Create base directories
    os.makedirs(output_detection_path, exist_ok=True)
    os.makedirs(output_detection_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    # Define splits
    splits = ["train", "val", "test"]
    total_copied = 0

    # Process each split
    for split in splits:
        # Create output directories for the split
        os.makedirs(os.path.join(output_images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(output_labels_dir, split), exist_ok=True)
        os.makedirs(os.path.join(output_masks_dir, split), exist_ok=True)

        # Paths for this split
        images_split_dir = os.path.join(detection_path, "images", split)
        labels_split_dir = os.path.join(detection_path, "labels", split)

        # Skip if split directories don't exist
        if not os.path.exists(images_split_dir) or not os.path.exists(labels_split_dir):
            print(f"Skipping {split} split - directories not found")
            continue

        # Get basenames (without extensions) for this split
        image_basenames = {
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(images_split_dir, "*.*"))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        }

        label_basenames = {
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(labels_split_dir, "*.txt"))
        }

        mask_basenames = {
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(masks_path, "*.*"))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        }

        # Find intersection
        common_basenames = image_basenames.intersection(label_basenames).intersection(
            mask_basenames
        )
        print(f"Found {len(common_basenames)} files in common for {split} split")
        total_copied += len(common_basenames)

        # Copy files
        for basename in common_basenames:
            # Find and copy image (might have different extensions)
            image_files = glob.glob(os.path.join(images_split_dir, f"{basename}.*"))
            for image_file in image_files:
                if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    shutil.copy2(
                        image_file,
                        os.path.join(
                            output_images_dir, split, os.path.basename(image_file)
                        ),
                    )
                    break

            # Copy label
            label_file = os.path.join(labels_split_dir, f"{basename}.txt")
            if os.path.exists(label_file):
                shutil.copy2(
                    label_file,
                    os.path.join(output_labels_dir, split, f"{basename}.txt"),
                )

            # Find and copy mask
            mask_files = glob.glob(os.path.join(masks_path, f"{basename}.*"))
            for mask_file in mask_files:
                if mask_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    shutil.copy2(
                        mask_file,
                        os.path.join(
                            output_masks_dir, split, os.path.basename(mask_file)
                        ),
                    )
                    break

    # Copy YAML file if provided
    if yaml_path and os.path.exists(yaml_path):
        shutil.copy2(
            yaml_path, os.path.join(output_detection_path, os.path.basename(yaml_path))
        )
        print(
            f"Copied YAML file to {os.path.join(output_detection_path, os.path.basename(yaml_path))}"
        )

    print(f"Dataset transformation complete. Output saved to {output_path}")
    print(f"- Copied {total_copied} matching files to the new dataset")


def main():
    transform_dataset(
        detection_path="/home/mariopasc/Python/Datasets/COMBINED/tasks/stenosis_detection/datasets/yolo",
        masks_path="/home/mariopasc/Python/Datasets/COMBINED/tasks/stenosis_detection/labels/masks",
        output_path="/home/mariopasc/Python/Datasets/COMBINED",
        yaml_path="/home/mariopasc/Python/Datasets/COMBINED/tasks/stenosis_detection/datasets/yolo/yolo_ica_detection.yaml",
    )


if __name__ == "__main__":
    main()
