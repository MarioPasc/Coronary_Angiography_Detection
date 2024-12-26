import os
import pandas as pd
from pathlib import Path
import shutil
import ast
from typing import Dict, List
from PIL import Image
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
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
IMAGE_DIR = Path(PROCESSED_DIR) / "images"
LABEL_DIR = Path(PROCESSED_DIR) / "labels"


# Utility functions
def _getImageDimensions(image_path: str):
    """Get image dimensions."""
    with Image.open(image_path) as img:
        return img.width, img.height


def _convertBboxFormatToYOLO(
    bbox: str, img_width: int, img_height: int, class_mappings: Dict[str, int]
) -> str:
    """Convert bounding box to YOLO format."""
    x, y, w, h, cls = bbox.split()
    x, y, w, h = int(x), int(y), int(w), int(h)
    cls = class_mappings.get(cls, 0)  # Map class to integer

    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w /= img_width
    h /= img_height

    return f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def preprocess_images_and_labels(
    df: pd.DataFrame, class_mappings: Dict[str, int]
) -> pd.DataFrame:
    """Copy and preprocess images and labels, updating their paths."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)

    updated_rows = []
    for _, row in df.iterrows():
        if row["Lesion"]:
            image_src = row["SelectedFramesLesionVideo"]
            label_src = row["GroundTruthFile"]
        else:
            image_src = row["SelectedFramesNonLesionVideo"]
            label_src = None

        # Destination paths
        image_dst = IMAGE_DIR / os.path.basename(image_src)
        label_dst = LABEL_DIR / os.path.basename(label_src) if label_src else None

        # Copy image
        if not image_dst.exists():
            shutil.copy(image_src, image_dst)

        # Process and copy label
        if label_src and label_src != "nolesion":
            try:
                img_width, img_height = _getImageDimensions(image_src)
                with open(label_src, "r") as f_in, open(label_dst, "w") as f_out:
                    for bbox in f_in:
                        yolo_bbox = _convertBboxFormatToYOLO(
                            bbox.strip(), img_width, img_height, class_mappings
                        )
                        f_out.write(yolo_bbox + "\n")
            except FileNotFoundError:
                print(f"Label file not found: {label_src}, skipping.")

        # Update row with new paths
        updated_row = row.copy()
        updated_row["SelectedFramesLesionVideo"] = str(image_dst)
        updated_row["SelectedFramesNonLesionVideo"] = str(image_dst)
        updated_row["GroundTruthFile"] = str(label_dst) if label_dst else "nolesion"
        updated_rows.append(updated_row)

    return pd.DataFrame(updated_rows)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(csv_path)


def run_filterByLabels(df: pd.DataFrame, labels_to_remove: List[str]) -> pd.DataFrame:
    """
    Filters the dataset by removing rows where the labels to remove are the only labels in LesionLabel.
    Updates LesionLabel and GroundTruthFile for rows with mixed labels.

    Args:
        df (pd.DataFrame): Input dataset containing LesionLabel and paths to ground truth files.
        labels_to_remove (List[str]): List of labels to remove.

    Returns:
        pd.DataFrame: Filtered and updated DataFrame.
    """

    def update_ground_truth_file(
        file_path: str, labels_to_remove: List[str]
    ) -> List[str]:
        """
        Updates the ground truth file by removing specified labels and returning the remaining labels.

        Args:
            file_path (str): Path to the ground truth file.
            labels_to_remove (List[str]): Labels to be removed.

        Returns:
            List[str]: Remaining labels after removal.
        """
        remaining_labels = set()
        try:
            if pd.isna(file_path) or file_path == "nolesion":
                return ["nolesion"]  # Ensure 'nolesion' images are preserved

            with open(file_path, "r") as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                label = line.strip().split()[
                    -1
                ]  # Extract the last element as the label
                if label not in labels_to_remove:
                    updated_lines.append(line)
                    remaining_labels.add(label)

            # Overwrite the file with updated content
            with open(file_path, "w") as f:
                f.writelines(updated_lines)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

        return list(remaining_labels)

    # Process each row in the DataFrame
    for idx, row in df.iterrows():
        if row["GroundTruthFile"] == "nolesion" or pd.isna(row["GroundTruthFile"]):
            df.at[idx, "LesionLabel"] = ["nolesion"]  # Mark explicitly
            continue  # Skip further processing for 'nolesion' entries

        # Update the ground truth file and get remaining labels
        remaining_labels = update_ground_truth_file(
            row["GroundTruthFile"], labels_to_remove
        )
        df.at[idx, "LesionLabel"] = remaining_labels  # Update the LesionLabel field

        # Remove row if no labels remain
        if not remaining_labels or remaining_labels == ["nolesion"]:
            df.drop(idx, inplace=True)

    return df


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Parse 'LesionLabel' strings into lists and handle nans."""
    df["LesionLabel"] = df["LesionLabel"].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) and x != "nan" else []
    )
    return df


def prepare_multilabel_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary multilabel matrix for patients.
    Args:
        df: Input DataFrame.
    Returns:
        DataFrame where each row corresponds to a patient and columns represent labels.
    """
    patients = df["Patient"].unique()
    patient_label_dict = {patient: set() for patient in patients}

    # Collect labels for each patient
    for _, row in df.iterrows():
        labels = row["LesionLabel"]
        patient_label_dict[row["Patient"]].update(labels)

    # Create binary multilabel DataFrame
    all_labels = sorted(set().union(*patient_label_dict.values()))
    label_matrix = pd.DataFrame(0, index=patients, columns=all_labels)
    for patient, labels in patient_label_dict.items():
        for label in labels:
            label_matrix.at[patient, label] = 1

    return label_matrix.reset_index().rename(columns={"index": "Patient"})


def save_multilabel_folds(
    df: pd.DataFrame,
    label_matrix: pd.DataFrame,
    output_dir: str,
    n_splits: int,
    internal_folds: int,
    seed: int,
):
    """
    Create external and internal folds using MultilabelStratifiedKFold and save splits.
    Args:
        df: Original DataFrame.
        label_matrix: Multilabel binary matrix for patients.
        output_dir: Output directory for folds.
        n_splits: Number of external folds.
        internal_folds: Number of internal folds for train/val split.
        seed: Random seed.
    """
    os.makedirs(output_dir, exist_ok=True)

    label_distribution = label_matrix.drop(columns=["Patient"]).sum(axis=0)
    print("Label Distribution:")
    print(label_distribution)

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    for fold_idx, (train_val_idx, test_idx) in enumerate(
        mskf.split(label_matrix["Patient"], label_matrix.drop(columns=["Patient"]))
    ):
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_dir, exist_ok=True)

        # Split patients
        train_val_patients = label_matrix.iloc[train_val_idx]["Patient"].tolist()
        test_patients = label_matrix.iloc[test_idx]["Patient"].tolist()

        test_df = df[df["Patient"].isin(test_patients)]
        train_val_df = df[df["Patient"].isin(train_val_patients)]

        test_df.to_csv(os.path.join(fold_dir, "test.csv"), index=False)
        logging.info(
            f"Saved Fold {fold_idx + 1}: Test patients count: {len(test_patients)}"
        )

        # Internal folds
        internal_dir = os.path.join(fold_dir, "internal_folds")
        os.makedirs(internal_dir, exist_ok=True)

        if internal_folds == 1:
            # Special logic when INTERNAL_FOLDS=1
            internal_fold_dir = os.path.join(internal_dir, "internal_fold_1")
            os.makedirs(internal_fold_dir, exist_ok=True)

            # Save the same train_val_df as train.csv and val.csv
            train_val_df.to_csv(
                os.path.join(internal_fold_dir, "train.csv"), index=False
            )
            train_val_df.to_csv(os.path.join(internal_fold_dir, "val.csv"), index=False)
            logging.info(
                f"Saved Internal Fold 1 for Fold {fold_idx + 1} with train and val identical."
            )

        else:
            # Standard logic when INTERNAL_FOLDS > 1
            internal_mskf = MultilabelStratifiedKFold(
                n_splits=internal_folds, random_state=seed, shuffle=True
            )
            train_val_labels = label_matrix[
                label_matrix["Patient"].isin(train_val_patients)
            ]

            for internal_idx, (train_idx, val_idx) in enumerate(
                internal_mskf.split(
                    train_val_labels["Patient"],
                    train_val_labels.drop(columns=["Patient"]),
                )
            ):
                internal_fold_dir = os.path.join(
                    internal_dir, f"internal_fold_{internal_idx + 1}"
                )
                os.makedirs(internal_fold_dir, exist_ok=True)

                train_patients = train_val_labels.iloc[train_idx]["Patient"].tolist()
                val_patients = train_val_labels.iloc[val_idx]["Patient"].tolist()

                train_df = df[df["Patient"].isin(train_patients)]
                val_df = df[df["Patient"].isin(val_patients)]

                train_df.to_csv(
                    os.path.join(internal_fold_dir, "train.csv"), index=False
                )
                val_df.to_csv(os.path.join(internal_fold_dir, "val.csv"), index=False)
                logging.info(
                    f"Saved Internal Fold {internal_idx + 1} for Fold {fold_idx + 1}"
                )


##########################
#    INTEGRITY CHECKS    #
##########################


def integrity_check_patients(output_dir):
    """
    Check that patients are only in their respective splits.
    Args:
        output_dir: Directory containing all outer and inner fold splits.
    """
    results = []
    for fold in os.listdir(output_dir):
        fold_path = os.path.join(output_dir, fold)
        if not os.path.isdir(fold_path):
            continue

        # Load test patients
        test_df = pd.read_csv(os.path.join(fold_path, "test.csv"))
        test_patients = set(test_df["Patient"])

        # Check internal folds
        internal_path = os.path.join(fold_path, "internal_folds")
        for internal_fold in os.listdir(internal_path):
            internal_fold_path = os.path.join(internal_path, internal_fold)
            train_df = pd.read_csv(os.path.join(internal_fold_path, "train.csv"))
            val_df = pd.read_csv(os.path.join(internal_fold_path, "val.csv"))
            train_patients = set(train_df["Patient"])
            val_patients = set(val_df["Patient"])

            # Verify no overlap
            overlap_test_train = test_patients.intersection(train_patients)
            overlap_test_val = test_patients.intersection(val_patients)
            overlap_train_val = train_patients.intersection(val_patients)

            results.append(
                [
                    fold,
                    internal_fold,
                    len(overlap_test_train),
                    len(overlap_test_val),
                    len(overlap_train_val),
                ]
            )

    # Save results to CSV
    results_df = pd.DataFrame(
        results,
        columns=[
            "OuterFold",
            "InnerFold",
            "Test_Train_Overlap",
            "Test_Val_Overlap",
            "Train_Val_Overlap",
        ],
    )
    results_df.to_csv(os.path.join(output_dir, "integrity_check.csv"), index=False)
    logging.info("Saved integrity check results.")


def label_distribution_check(output_dir, labels):
    """
    Create label distribution statistics for each fold.
    Args:
        output_dir: Directory containing all outer and inner fold splits.
        labels: List of labels to check.
    """
    results = []
    for fold in os.listdir(output_dir):
        fold_path = os.path.join(output_dir, fold)
        if not os.path.isdir(fold_path):
            continue

        # Check test split
        test_df = pd.read_csv(os.path.join(fold_path, "test.csv"))
        label_counts = {
            label: sum(test_df["LesionLabel"].apply(lambda x: str(label) in str(x)))
            for label in labels
        }
        results.append(["test", fold, *label_counts.values()])

        # Check internal folds
        internal_path = os.path.join(fold_path, "internal_folds")
        for internal_fold in os.listdir(internal_path):
            internal_fold_path = os.path.join(internal_path, internal_fold)
            for split in ["train", "val"]:
                split_df = pd.read_csv(os.path.join(internal_fold_path, f"{split}.csv"))
                label_counts = {
                    label: sum(
                        split_df["LesionLabel"].apply(lambda x: str(label) in str(x))
                    )
                    for label in labels
                }
                results.append(
                    [split, f"{fold}_{internal_fold}", *label_counts.values()]
                )

    # Save results to CSV
    columns = ["Split", "Fold"] + labels
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv(os.path.join(output_dir, "label_distribution.csv"), index=False)
    logging.info("Saved label distribution results.")


if __name__ == "__main__":
    # Configuration
    CSV_PATH = CONFIG["ORIGINAL_INFORMATION_DATASET_CSV_PATH"]
    OUTPUT_DIR = CONFIG["SPLITS_CSV_DIR"]
    EXTERNAL_FOLDS = CONFIG["EXTERNAL_FOLDS"]
    SEED = CONFIG["SEED"]
    LABELS_TO_REMOVE = CONFIG["LABELS_TO_REMOVE"]
    INTERNAL_FOLDS = CONFIG["INTERNAL_FOLDS"]
    CLASS_MAPPINGS = CONFIG["CLASS_MAPPINGS"]

    # Dynamic variables
    LABELS = list(set(CLASS_MAPPINGS.keys()) - set(LABELS_TO_REMOVE))
    OUTPUT_PROCESSED_CSV = os.path.join(
        CONFIG["PROCESSED_DIR"], "information_dataset_processed.csv"
    )

    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True,
    )

    # Load and preprocess data
    df = load_data(CSV_PATH)
    logging.info(f"Loaded dataset CSV with {len(df)} samples.")
    df = run_filterByLabels(df, LABELS_TO_REMOVE)
    logging.info(
        f"Filtered dataset has {len(df)} samples after removing labels: {LABELS_TO_REMOVE}"
    )
    processed_df = preprocess_images_and_labels(df, CLASS_MAPPINGS)
    logging.info("Image and label preprocessing complete.")

    # Step 2: Save updated CSV
    processed_df.to_csv(OUTPUT_PROCESSED_CSV, index=False)
    logging.info(f"Saved updated dataset to {OUTPUT_PROCESSED_CSV}.")

    df = processed_df

    # df = pd.read_csv(OUTPUT_PROCESSED_CSV)

    # Prepare multilabel data
    label_matrix = prepare_multilabel_data(df)
    logging.info("Created patient-label binary matrix for stratified splitting.")

    # Save folds
    save_multilabel_folds(
        df, label_matrix, OUTPUT_DIR, EXTERNAL_FOLDS, INTERNAL_FOLDS, SEED
    )
    logging.info(
        "Saved all external and internal folds using MultilabelStratifiedKFold."
    )

    # Run integrity check
    integrity_check_patients(OUTPUT_DIR)
    logging.info("Integrity check complete.")

    # Run label distribution check
    label_distribution_check(OUTPUT_DIR, LABELS)
    logging.info("Label distribution check complete.")
