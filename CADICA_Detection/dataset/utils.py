# utils.py

import pandas as pd
from collections import Counter

def run_compute_augmentation_counts(csv_path: str, labels_to_exclude: list, total_images_to_augment: int, output_csv_path: str) -> None:
    """
    Computes the number of augmented images needed per label to balance the dataset.

    Args:
    -----
    csv_path : str
        Path to the CSV file containing the dataset.
    labels_to_exclude : list
        List of labels that should not be augmented.
    total_images_to_augment : int
        Total number of images the user wants to augment.
    output_csv_path : str
        Path to save the output CSV file with augmentation counts.

    Returns:
    --------
    None
    """
    # Read and Parse the Dataset CSV
    df = pd.read_csv(csv_path)

    # Parse 'LesionLabel' column to lists, keeping duplicates
    def parse_labels(label_str):
        if pd.isna(label_str) or label_str == 'nolesion':
            return []
        else:
            return label_str.split(',')

    df['LesionLabelList'] = df['LesionLabel'].apply(parse_labels)

    # Flatten the list of labels and count frequencies, including duplicates
    all_labels = []
    for labels in df['LesionLabelList']:
        all_labels.extend(labels)
    label_counts = Counter(all_labels)

    # Compute Original Counts
    counts_df = pd.DataFrame.from_dict(label_counts, orient='index', columns=['Original_Counts'])
    counts_df.index.name = 'Label'
    counts_df.reset_index(inplace=True)

    # Determine Labels to Augment
    # Exclude labels specified by the user
    labels_to_include = counts_df[~counts_df['Label'].isin(labels_to_exclude)]['Label'].tolist()
    counts_df = counts_df[counts_df['Label'].isin(labels_to_include)]

    if counts_df.empty:
        print("No labels to augment after excluding specified labels.")
        return

    # Calculate Target Counts
    max_count = counts_df['Original_Counts'].max()
    counts_df['Target_Count'] = max_count

    # Compute Required Increase per Label
    counts_df['Required_Increase'] = counts_df['Target_Count'] - counts_df['Original_Counts']

    # Adjust for Total Images to Augment
    total_required_increase = counts_df['Required_Increase'].sum()
    if total_required_increase == 0:
        print("All labels are already balanced.")
        counts_df['Augmented_Counts'] = 0
    else:
        # Scale Required_Increase to match total_images_to_augment
        scaling_factor = total_images_to_augment / total_required_increase
        counts_df['Augmented_Counts'] = (counts_df['Required_Increase'] * scaling_factor).round().astype(int)

    # Output CSV File
    output_df = counts_df[['Label', 'Original_Counts', 'Augmented_Counts']]
    output_df.to_csv(output_csv_path, index=False)
    print(f"Augmentation counts saved to {output_csv_path}")