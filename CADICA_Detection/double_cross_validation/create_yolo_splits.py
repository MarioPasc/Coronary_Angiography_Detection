import os
import yaml
import pandas as pd
from pathlib import Path
import shutil
import logging


def load_config(config_path: str):
    """Load configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def copy_or_link_file(src, dst, use_symlinks):
    """
    Copy or create symbolic link for a file based on configuration.

    Args:
        src (str): Source file path.
        dst (str): Destination file path.
        use_symlinks (bool): If True, create symbolic links; else, copy files.
    """
    if not os.path.exists(dst):
        if use_symlinks:
            os.symlink(src, dst)
        else:
            shutil.copy(src, dst)


def create_yolo_files(csv_path, output_dir, split_type, use_symlinks):
    """
    Create YOLO-compatible datasets (images and labels) with either symbolic links or copies.

    Args:
        csv_path (str): Path to the CSV file.
        output_dir (str): Directory to store files.
        split_type (str): Split type ('train', 'val', 'test').
        use_symlinks (bool): Flag to determine if symbolic links or copies should be created.
    """
    df = pd.read_csv(csv_path)
    image_output_dir = Path(output_dir) / "images" / split_type
    label_output_dir = Path(output_dir) / "labels" / split_type

    # Ensure directories exist
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    # Iterate through rows and copy or link files
    for _, row in df.iterrows():
        image_src = (
            row["SelectedFramesLesionVideo"] or row["SelectedFramesNonLesionVideo"]
        )
        label_src = row["GroundTruthFile"]

        # Define destination paths
        image_dst = image_output_dir / os.path.basename(image_src)
        label_dst = (
            label_output_dir / os.path.basename(label_src)
            if label_src != "nolesion"
            else None
        )

        # Copy or link files
        copy_or_link_file(image_src, image_dst, use_symlinks)
        if label_dst:
            copy_or_link_file(label_src, label_dst, use_symlinks)


def generate_yolo_splits(root_folder, output_dir, use_symlinks):
    """
    Generate YOLO-compatible datasets for each fold.

    Args:
        root_folder (str): Path to the root folder containing splits.
        output_dir (str): Directory to store YOLO datasets.
        use_symlinks (bool): Flag to determine if symbolic links or copies should be created.
    """
    for outer_fold in sorted(os.listdir(root_folder)):
        outer_fold_path = Path(root_folder) / outer_fold
        if not outer_fold_path.is_dir():
            continue

        print(f"Processing {outer_fold}...")
        test_csv = outer_fold_path / "test.csv"

        # Process internal folds
        internal_folds_path = outer_fold_path / "internal_folds"
        for inner_fold in sorted(os.listdir(internal_folds_path)):
            inner_fold_path = internal_folds_path / inner_fold
            if not inner_fold_path.is_dir():
                continue

            print(f"  Processing {inner_fold}...")
            inner_output_dir = Path(output_dir) / outer_fold / inner_fold

            # Create YOLO files
            create_yolo_files(test_csv, inner_output_dir, "test", use_symlinks)
            create_yolo_files(
                inner_fold_path / "train.csv", inner_output_dir, "train", use_symlinks
            )
            create_yolo_files(
                inner_fold_path / "val.csv", inner_output_dir, "val", use_symlinks
            )

    print("YOLO-compatible datasets have been generated successfully!")


def generate_fold_configs(root_folder: str, output_base_dir: str) -> None:
    """
    Generate YOLO configuration files for each outer/inner fold.

    Args:
        root_folder (str): Path to the root folder containing outer folds.
        output_base_dir (str): Directory where configuration files will be saved.
    """
    os.makedirs(output_base_dir, exist_ok=True)

    # Iterate over outer folds
    for outer_idx, outer_fold in enumerate(sorted(os.listdir(root_folder)), start=1):
        outer_fold_path = Path(root_folder) / outer_fold
        if not outer_fold_path.is_dir():
            continue

        internal_folds_path = outer_fold_path / "internal_folds"
        for inner_idx, inner_fold in enumerate(
            sorted(os.listdir(internal_folds_path)), start=1
        ):
            inner_fold_path = os.path.join(output_base_dir, f"fold_{outer_idx}", f"internal_fold_{inner_idx}")
            # Create YAML configuration
            config = {
                "path": str(inner_fold_path),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "lesion"},
            }

            # Generate a unique config file name
            config_filename = f"outer_{outer_idx}_inner_{inner_idx}.yaml"
            config_path = Path(output_base_dir) / config_filename

            # Save the configuration
            with open(config_path, "w") as file:
                yaml.dump(config, file, default_flow_style=False)

            logging.info(f"Created config file: {config_path}")


def main():
    # Load configuration
    CONFIG = load_config("cv_config.yaml")

    # Extract relevant variables
    root_folder = CONFIG["SPLITS_CSV_DIR"]
    output_yolo_dir = CONFIG["OUTPUT_YOLO_DIR"]
    use_symlinks = CONFIG["SYMBOLIC_LINKS"]

    # Generate YOLO splits
    generate_yolo_splits(root_folder, output_yolo_dir, use_symlinks)
    generate_fold_configs(root_folder, output_yolo_dir)


if __name__ == "__main__":
    main()
