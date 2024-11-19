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


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import seaborn as sns


    # Set up argument parser
    parser = argparse.ArgumentParser(description="Compute augmentation counts and plot label distributions.")
    parser.add_argument('--train_csv', type=str, required=True, help='Path to train.csv')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to val.csv')
    parser.add_argument('--labels_to_exclude', nargs='*', default=[], help='List of labels not to augment')
    parser.add_argument('--total_images_to_augment_train', type=int, required=True, help='Total images to augment in train set')
    parser.add_argument('--total_images_to_augment_val', type=int, required=True, help='Total images to augment in val set')
    parser.add_argument('--output_csv_train', type=str, default='train_augmentation_counts.csv', help='Output CSV for train augmentation counts')
    parser.add_argument('--output_csv_val', type=str, default='val_augmentation_counts.csv', help='Output CSV for val augmentation counts')
    args = parser.parse_args()

    # Compute augmentation counts for train set
    compute_augmentation_counts(
        csv_path=args.train_csv,
        labels_to_exclude=args.labels_to_exclude,
        total_images_to_augment=args.total_images_to_augment_train,
        output_csv_path=args.output_csv_train
    )

    # Compute augmentation counts for validation set
    compute_augmentation_counts(
        csv_path=args.val_csv,
        labels_to_exclude=args.labels_to_exclude,
        total_images_to_augment=args.total_images_to_augment_val,
        output_csv_path=args.output_csv_val
    )

    # Read the original datasets
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    # Parse 'LesionLabel' column to lists, keeping duplicates
    def parse_labels(label_str):
        if pd.isna(label_str) or label_str == 'nolesion':
            return []
        else:
            return label_str.split(',')

    train_df['LesionLabelList'] = train_df['LesionLabel'].apply(parse_labels)
    val_df['LesionLabelList'] = val_df['LesionLabel'].apply(parse_labels)

    # Compute original label counts
    all_labels_train = []
    for labels in train_df['LesionLabelList']:
        all_labels_train.extend(labels)
    label_counts_train = Counter(all_labels_train)
    counts_train_df = pd.DataFrame.from_dict(label_counts_train, orient='index', columns=['Original_Counts'])
    counts_train_df.index.name = 'Label'
    counts_train_df.reset_index(inplace=True)

    all_labels_val = []
    for labels in val_df['LesionLabelList']:
        all_labels_val.extend(labels)
    label_counts_val = Counter(all_labels_val)
    counts_val_df = pd.DataFrame.from_dict(label_counts_val, orient='index', columns=['Original_Counts'])
    counts_val_df.index.name = 'Label'
    counts_val_df.reset_index(inplace=True)

    # Read augmentation counts
    aug_counts_train_df = pd.read_csv(args.output_csv_train)
    aug_counts_val_df = pd.read_csv(args.output_csv_val)

    # Compute new counts after augmentation
    aug_counts_train_df['New_Counts'] = aug_counts_train_df['Original_Counts'] + aug_counts_train_df['Augmented_Counts']
    aug_counts_val_df['New_Counts'] = aug_counts_val_df['Original_Counts'] + aug_counts_val_df['Augmented_Counts']

    # Merge with original counts to include labels not augmented
    full_counts_train_df = counts_train_df.merge(
        aug_counts_train_df[['Label', 'Augmented_Counts', 'New_Counts']],
        on='Label',
        how='left'
    )
    full_counts_train_df['Augmented_Counts'] = full_counts_train_df['Augmented_Counts'].fillna(0)
    full_counts_train_df['New_Counts'] = full_counts_train_df['New_Counts'].fillna(full_counts_train_df['Original_Counts'])

    full_counts_val_df = counts_val_df.merge(
        aug_counts_val_df[['Label', 'Augmented_Counts', 'New_Counts']],
        on='Label',
        how='left'
    )
    full_counts_val_df['Augmented_Counts'] = full_counts_val_df['Augmented_Counts'].fillna(0)
    full_counts_val_df['New_Counts'] = full_counts_val_df['New_Counts'].fillna(full_counts_val_df['Original_Counts'])

    # Plot for train set
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Label', y='Original_Counts', data=full_counts_train_df, color='blue', label='Original Counts')
    sns.barplot(x='Label', y='Augmented_Counts', data=full_counts_train_df, bottom=full_counts_train_df['Original_Counts'], color='orange', label='Augmented Counts')
    plt.xticks(rotation=45)
    plt.xlabel('Labels')
    plt.ylabel('Counts')
    plt.title('Label Counts Before and After Augmentation (Train Set)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot for validation set
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Label', y='Original_Counts', data=full_counts_val_df, color='blue', label='Original Counts')
    sns.barplot(x='Label', y='Augmented_Counts', data=full_counts_val_df, bottom=full_counts_val_df['Original_Counts'], color='orange', label='Augmented Counts')
    plt.xticks(rotation=45)
    plt.xlabel('Labels')
    plt.ylabel('Counts')
    plt.title('Label Counts Before and After Augmentation (Validation Set)')
    plt.legend()
    plt.tight_layout()
    plt.show()
