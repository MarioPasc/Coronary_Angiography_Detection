import os
import pandas as pd
from sklearn.model_selection import KFold
import logging
from typing import List, Dict
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import ast


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(csv_path: str) -> pd.DataFrame:
    """Load dataset from the CSV file."""
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
    def update_ground_truth_file(file_path: str, labels_to_remove: List[str]) -> List[str]:
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
                label = line.strip().split()[-1]  # Extract the last element as the label
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
        remaining_labels = update_ground_truth_file(row["GroundTruthFile"], labels_to_remove)
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
    patients = df['Patient'].unique()
    patient_label_dict = {patient: set() for patient in patients}

    # Collect labels for each patient
    for _, row in df.iterrows():
        labels = row['LesionLabel']
        patient_label_dict[row['Patient']].update(labels)

    # Create binary multilabel DataFrame
    all_labels = sorted(set().union(*patient_label_dict.values()))
    label_matrix = pd.DataFrame(0, index=patients, columns=all_labels)
    for patient, labels in patient_label_dict.items():
        for label in labels:
            label_matrix.at[patient, label] = 1

    return label_matrix.reset_index().rename(columns={'index': 'Patient'})

def save_multilabel_folds(df: pd.DataFrame, label_matrix: pd.DataFrame, output_dir: str, n_splits: int, internal_folds: int, seed: int):
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

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    for fold_idx, (train_val_idx, test_idx) in enumerate(mskf.split(label_matrix['Patient'], label_matrix.drop(columns=['Patient']))):
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_dir, exist_ok=True)

        # Split patients
        train_val_patients = label_matrix.iloc[train_val_idx]['Patient'].tolist()
        test_patients = label_matrix.iloc[test_idx]['Patient'].tolist()

        test_df = df[df['Patient'].isin(test_patients)]
        train_val_df = df[df['Patient'].isin(train_val_patients)]

        test_df.to_csv(os.path.join(fold_dir, "test.csv"), index=False)
        train_val_df.to_csv(os.path.join(fold_dir, "train_val.csv"), index=False)
        logging.info(f"Saved Fold {fold_idx + 1}: Test patients count: {len(test_patients)}")

        # Internal folds
        internal_dir = os.path.join(fold_dir, "internal_folds")
        os.makedirs(internal_dir, exist_ok=True)

        internal_mskf = MultilabelStratifiedKFold(n_splits=internal_folds, random_state=seed, shuffle=True)
        train_val_labels = label_matrix[label_matrix['Patient'].isin(train_val_patients)]
        
        for internal_idx, (train_idx, val_idx) in enumerate(internal_mskf.split(train_val_labels['Patient'], train_val_labels.drop(columns=['Patient']))):
            internal_fold_dir = os.path.join(internal_dir, f"internal_fold_{internal_idx + 1}")
            os.makedirs(internal_fold_dir, exist_ok=True)

            train_patients = train_val_labels.iloc[train_idx]['Patient'].tolist()
            val_patients = train_val_labels.iloc[val_idx]['Patient'].tolist()

            train_df = df[df['Patient'].isin(train_patients)]
            val_df = df[df['Patient'].isin(val_patients)]

            train_df.to_csv(os.path.join(internal_fold_dir, "train.csv"), index=False)
            val_df.to_csv(os.path.join(internal_fold_dir, "val.csv"), index=False)
            logging.info(f"Saved Internal Fold {internal_idx + 1} for Fold {fold_idx + 1}")

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
        test_patients = set(test_df['Patient'])
        
        # Check internal folds
        internal_path = os.path.join(fold_path, "internal_folds")
        for internal_fold in os.listdir(internal_path):
            internal_fold_path = os.path.join(internal_path, internal_fold)
            train_df = pd.read_csv(os.path.join(internal_fold_path, "train.csv"))
            val_df = pd.read_csv(os.path.join(internal_fold_path, "val.csv"))
            train_patients = set(train_df['Patient'])
            val_patients = set(val_df['Patient'])
            
            # Verify no overlap
            overlap_test_train = test_patients.intersection(train_patients)
            overlap_test_val = test_patients.intersection(val_patients)
            overlap_train_val = train_patients.intersection(val_patients)
            
            results.append([fold, internal_fold, len(overlap_test_train), len(overlap_test_val), len(overlap_train_val)])
    
    # Save results to CSV
    results_df = pd.DataFrame(results, columns=["OuterFold", "InnerFold", "Test_Train_Overlap", "Test_Val_Overlap", "Train_Val_Overlap"])
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
        label_counts = {label: sum(test_df['LesionLabel'].apply(lambda x: str(label) in str(x))) for label in labels}
        results.append(["test", fold, *label_counts.values()])
        
        # Check internal folds
        internal_path = os.path.join(fold_path, "internal_folds")
        for internal_fold in os.listdir(internal_path):
            internal_fold_path = os.path.join(internal_path, internal_fold)
            for split in ["train", "val"]:
                split_df = pd.read_csv(os.path.join(internal_fold_path, f"{split}.csv"))
                label_counts = {label: sum(split_df['LesionLabel'].apply(lambda x: str(label) in str(x))) for label in labels}
                results.append([split, f"{fold}_{internal_fold}", *label_counts.values()])
    
    # Save results to CSV
    columns = ["Split", "Fold"] + labels
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv(os.path.join(output_dir, "label_distribution.csv"), index=False)
    logging.info("Saved label distribution results.")

if __name__ == "__main__":
    # Configuration
    CSV_PATH = "/home/mario/Python/Datasets/CADICA_Project/information_dataset.csv"  # Input CSV path
    OUTPUT_DIR = "./double_cv_splits"       # Output directory for folds
    EXTERNAL_FOLDS = 5                       # Number of external folds
    SEED = 42                                # Random seed for reproducibility
    LABELS = ["p100", "p99", "p90_98", "p70_90", "p50_70"]  # Labels to check
    LABELS_TO_REMOVE = ["p0_20", "p20_50"]  # Labels to remove
    INTERNAL_FOLDS = 3
    
    # Load and preprocess data
    df = load_data(CSV_PATH)
    logging.info(f"Loaded dataset CSV with {len(df)} samples.")
    df = run_filterByLabels(df, LABELS_TO_REMOVE)
    logging.info(f"Filtered dataset has {len(df)} samples after removing labels: {LABELS_TO_REMOVE}")

    # Prepare multilabel data
    label_matrix = prepare_multilabel_data(df)
    logging.info("Created patient-label binary matrix for stratified splitting.")

    # Save folds
    save_multilabel_folds(df, label_matrix, OUTPUT_DIR, EXTERNAL_FOLDS, INTERNAL_FOLDS, SEED)
    logging.info("Saved all external and internal folds using MultilabelStratifiedKFold.")
    
    # Run integrity check
    integrity_check_patients(OUTPUT_DIR)
    logging.info("Integrity check complete.")
    
    # Run label distribution check
    label_distribution_check(OUTPUT_DIR, LABELS)
    logging.info("Label distribution check complete.")

