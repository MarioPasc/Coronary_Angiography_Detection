import os
import pandas as pd
from pathlib import Path
import logging
import yaml

def load_config(config_path: str):
    """Load configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

CONFIG = load_config("./cv_config.yaml")

# Global variables for processed paths
PROCESSED_DIR = CONFIG["PROCESSED_DIR"]
SPLITS_CSV_DIR = CONFIG["SPLITS_CSV_DIR"]

# Function to transform the dataset format
def transform_dataset_format(input_csv: str, output_csv: str):
    """
    Transform the input dataset format into the expected format.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the transformed CSV file.
    """
    try:
        df = pd.read_csv(input_csv)

        # Map columns to the expected format
        transformed_df = pd.DataFrame({
            "LesionLabel": df["LesionLabel"].apply(lambda x: x[2:-2] if pd.notnull(x) else "nolesion"),
            "Frame_path": df["SelectedFramesLesionVideo"].combine_first(df["SelectedFramesNonLesionVideo"]),
            "Groundtruth_path": df["GroundTruthFile"].apply(lambda x: x if pd.notnull(x) else "nolesion"),
        })

        # Save the transformed dataset
        transformed_df.to_csv(output_csv, index=False)
        logging.info(f"Transformed dataset saved to {output_csv}")

    except Exception as e:
        logging.error(f"Error transforming dataset {input_csv}: {e}")

# Iterate through each fold and generate the transformed CSV files
def generate_transformed_folds():
    """
    Transform train, val, and test CSV files for each fold into the expected format.
    """
    for fold_dir in Path(SPLITS_CSV_DIR).iterdir():
        if not fold_dir.is_dir():
            continue

        test_csv = fold_dir / "test.csv"
        internal_folds_dir = fold_dir / "internal_folds"

        for internal_fold_dir in internal_folds_dir.iterdir():
            if not internal_fold_dir.is_dir():
                continue

            train_csv = internal_fold_dir / "train.csv"
            val_csv = internal_fold_dir / "val.csv"

            # Transform train, val, and test datasets
            output_train_csv = internal_fold_dir / "train.csv"
            output_val_csv = internal_fold_dir / "val.csv"
            output_test_csv = fold_dir / "test.csv"

            transform_dataset_format(train_csv, output_train_csv)
            transform_dataset_format(val_csv, output_val_csv)
            transform_dataset_format(test_csv, output_test_csv)

if __name__ == "__main__":
    generate_transformed_folds()