# Ideas for visualization in the dataset:
# 1. Samples per set (training/validation/test). Include a subplot with 2 plots: 
#       I. Set-based visualization: barplot; X-axis: "training" "validation" "test". Y-axis: counts; bars: "Lesion" "No Lesion" "Total" ✅ 
#       II. Label-based visualization: barplot; X-axis: Diagnostic labels, Y-axis: counts; bars: "Training" "Validation" "Test" ✅ 
# 2. Undersampling/Augmentation types used.
#       I. 4 images from a specific patient that show each augmentation image generated ✅ 
#       II. Augmented images per label: Comparison between original, undersampled, and augmented ✅ 
# 3. Simulated Annealing visualization: One scatterplot per adjusted hyperparameter, mark the best result. X-Axis: hyperparameter values; Y-Axis: fitness
# 4. Goal-oriented sensibility study of hyperparameters. 
#       I. lineplot for variation of mAP@50-95 per epoch for every hyperparameter value tried. X-axis: epoch, Y-axis: mAP@50-95, legend: Hyperparameter values
#       II. lineplot for variation of mAP@50-95 per hyperparameter value. X-axis: hyperparameter value, Y-axis: mAP@50-95
# 5. Results comparison: Lineplot of mAP@50-95 performance per model. X-axis: epoch, Y-axis: mAP@50-95, one data series per model (baseline, Simulated Annealing, iteration 1, iteration 2)

from typing import List
import pandas as pd
import os
import logging
import json
from collections import defaultdict
import re
import yaml
import cv2
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'ieee', 'std-colors'])
plt.rcParams['font.size'] = 12
plt.rcParams.update({'figure.dpi': '100'})
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

CONFIG_PATH = "./scripts/config.yaml"

# Set up logging
logging.basicConfig(filename='visualization.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting plots generation process.")

def load_config(yaml_path: str) -> dict:
    """
    Loads configuration parameters from a YAML file into a dictionary.
    
    Args
    -----
    yaml_path : str
        Path to the YAML configuration file.
    
    Returns
    -----
    dict
        Configuration parameters loaded from YAML.
    """
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error("Error loading config.yaml file; Are you executing the script from the root folder? If so, check this .py and change CONFIG_PATH.")

# Load the CONFIG variable
CONFIG = load_config(CONFIG_PATH)

def original_dataset_visualization():
    """
    Generates and saves visualizations for original dataset analysis: 
    1) Samples per set (training/validation/test) for Lesion, No Lesion, and Total counts.
    2) Label-based counts for diagnostic labels in each set.

    Args
    ----
    original_folder : str
        Path to the folder containing original train.csv, val.csv, test.csv files.
    save_path : str
        Path where the figures will be saved.
    save_formats : List[str], optional
        List of formats in which to save the figures (default is ['png']).
    """
    try:
        # Load variables
        original_folder = os.path.join(CONFIG["OUTPUT_PATH"], "CADICA_Holdout_Info")
        save_path = os.path.join(CONFIG["OUTPUT_PATH"], "PAPER_Figures")
        os.makedirs(save_path, exist_ok=True)
        save_formats = CONFIG["FIGURE_FORMATS"]

        # Load the original datasets
        original_train = pd.read_csv(os.path.join(original_folder, 'train.csv'))
        original_val = pd.read_csv(os.path.join(original_folder, 'val.csv'))
        original_test = pd.read_csv(os.path.join(original_folder, 'test.csv'))
        
        # Helper function to categorize lesion and non-lesion counts for each set
        def categorize_lesion_counts(df, set_name):
            lesion_count = len(df[df['LesionLabel'] != 'nolesion'])
            no_lesion_count = len(df[df['LesionLabel'] == 'nolesion'])
            total_count = len(df)
            return {'Set': set_name, 'Lesion': lesion_count, 'No Lesion': no_lesion_count, 'Total': total_count}

        # Calculate counts for each set (train, val, test)
        counts_per_set = pd.DataFrame([
            categorize_lesion_counts(original_train, 'Training'),
            categorize_lesion_counts(original_val, 'Validation'),
            categorize_lesion_counts(original_test, 'Test')
        ])

        # Define desired label order
        label_order = ["nolesion", "p0_20", "p20_50", "p50_70", "p70_90", "p90_98", "p99", "p100"]

        # Calculate label-based counts for each set in specified order
        label_counts = pd.DataFrame({
            'Training': original_train['LesionLabel'].value_counts(),
            'Validation': original_val['LesionLabel'].value_counts(),
            'Test': original_test['LesionLabel'].value_counts()
        }).reindex(label_order).fillna(0).astype(int)

        # Determine the max value across both plots to standardize y-axis
        max_y = max(counts_per_set[['Lesion', 'No Lesion', 'Total']].values.max(), label_counts.values.max())

        # Plotting with adjusted y-axis
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Set-based visualization with standardized y-axis
        counts_per_set.plot(kind='bar', x='Set', y=['Lesion', 'No Lesion', 'Total'], ax=axs[0])
        axs[0].set_title("Sample Counts per Set")
        axs[0].set_ylabel("Counts")
        axs[0].set_xlabel("Original Set")
        axs[0].set_ylim(0, max_y)
        axs[0].tick_params(axis='x', rotation=0)  # Rotate x-axis labels

        # Plot 2: Label-based visualization with standardized y-axis
        label_counts.plot(kind='bar', ax=axs[1])
        axs[1].set_title("Label-based Sample Counts")
        #axs[1].set_ylabel("Counts")
        axs[1].set_xlabel("Original Diagnostic Labels")
        axs[1].set_ylim(0, max_y)
        axs[1].set_yticklabels([])  # Disable y-axis numbers
        axs[1].tick_params(axis='x', rotation=0)  # Rotate x-axis labels

        for ax in axs:
            ax.spines[['right', 'top']].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        plt.tight_layout()

        # Save figures in specified formats
        for fmt in save_formats:
            fig.savefig(os.path.join(save_path, f'dataset_visualizations.{fmt}'), format=fmt)

        plt.show()
        logging.info(f"Original dataset plots generates and saved in {save_path}.")
    except Exception as e:
        logging.info(f"Error generating original dataset plots. - {e}")

def processed_dataset_visualization():
    """
    Generates and saves visualizations for processed dataset analysis, showing changes in counts due to augmentation and undersampling.
    Adds markers to each bar representing the original dataset counts for comparison.

    Args
    ----
    No direct arguments, loads paths and parameters from CONFIG.
    """
    try:
        # Load variables
        original_folder = os.path.join(CONFIG["OUTPUT_PATH"], "CADICA_Holdout_Info")
        processed_folder = os.path.join(CONFIG["OUTPUT_PATH"], "CADICA_Augmented_Images")
        processed_test_folder = os.path.join(CONFIG["OUTPUT_PATH"], "CADICA_Undersampled_Images")
        save_path = os.path.join(CONFIG["OUTPUT_PATH"], "PAPER_Figures")
        os.makedirs(save_path, exist_ok=True)
        save_formats = CONFIG["FIGURE_FORMATS"]

        # Load original datasets
        original_train = pd.read_csv(os.path.join(original_folder, 'train.csv'))
        original_val = pd.read_csv(os.path.join(original_folder, 'val.csv'))
        original_test = pd.read_csv(os.path.join(original_folder, 'test.csv'))

        # Load processed datasets
        processed_train = pd.read_csv(os.path.join(processed_folder, 'processed_train.csv'))
        processed_val = pd.read_csv(os.path.join(processed_folder, 'processed_val.csv'))
        processed_test = pd.read_csv(os.path.join(processed_test_folder, 'processed_test.csv'))

        # Helper function to categorize lesion and non-lesion counts for each set
        def categorize_counts(df, set_name):
            lesion_count = len(df[df['LesionLabel'] != 'nolesion'])
            no_lesion_count = len(df[df['LesionLabel'] == 'nolesion'])
            total_count = len(df)
            return {'Set': set_name, 'Lesion': lesion_count, 'No Lesion': no_lesion_count, 'Total': total_count}

        # Calculate original and processed counts for each set
        original_counts = pd.DataFrame([
            categorize_counts(original_train, 'Training'),
            categorize_counts(original_val, 'Validation'),
            categorize_counts(original_test, 'Test')
        ])

        processed_counts = pd.DataFrame([
            categorize_counts(processed_train, 'Training'),
            categorize_counts(processed_val, 'Validation'),
            categorize_counts(processed_test, 'Test')
        ])

        # Define desired label order
        label_order = ["nolesion", "p0_20", "p20_50", "p50_70", "p70_90", "p90_98", "p99", "p100"]

        # Calculate original and processed label-based counts
        def label_counts(df, label_order, label_name):
            return pd.DataFrame({label_name: df['LesionLabel'].value_counts()}).reindex(label_order).fillna(0).astype(int)

        original_labels = pd.concat([
            label_counts(original_train, label_order, 'Training'),
            label_counts(original_val, label_order, 'Validation'),
            label_counts(original_test, label_order, 'Test')
        ], axis=1)

        processed_labels = pd.concat([
            label_counts(processed_train, label_order, 'Training'),
            label_counts(processed_val, label_order, 'Validation'),
            label_counts(processed_test, label_order, 'Test')
        ], axis=1)

        # Determine max y-axis limit
        max_y = max(processed_counts[['Lesion', 'No Lesion', 'Total']].values.max(), processed_labels.values.max())

        # Plotting with adjusted y-axis
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Processed set-based visualization with markers for original values
        offset = 0.165
        processed_counts.plot(kind='bar', x='Set', y=['Lesion', 'No Lesion', 'Total'], ax=axs[0])
        for i, row in original_counts.iterrows():
            axs[0].plot([i - offset, i, i + offset], row[['Lesion', 'No Lesion', 'Total']], "^", color="red", markersize = 5, 
                        label="Original Count" if i == 0 else "")
        axs[0].set_title("Processed Sample Counts per Set")
        axs[0].set_ylabel("Counts")
        axs[0].set_ylim(0, max_y)
        axs[0].legend()
        axs[0].tick_params(axis='x', rotation=0)  # Rotate x-axis labels

        # Plot 2: Processed label-based visualization with markers for original values
        processed_labels.plot(kind='bar', ax=axs[1])
        for i, label in enumerate(label_order):
            axs[1].plot([i - offset, i, i + offset], original_labels.loc[label, ['Training', 'Validation', 'Test']], "x", color="red", markersize=5, 
                        label="Original Count" if i == 0 else "")
        axs[1].set_title("Processed Label-based Sample Counts")
        axs[1].set_ylim(0, max_y)
        axs[1].set_yticklabels([])  # Disable y-axis numbers
        axs[1].legend()
        axs[1].tick_params(axis='x', rotation=0)  # Rotate x-axis labels

        for ax in axs:
            ax.spines[['right', 'top']].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
        plt.tight_layout()

        # Save figures in specified formats
        for fmt in save_formats:
            fig.savefig(os.path.join(save_path, f'processed_dataset_visualizations.{fmt}'), format=fmt)

        plt.show()
        logging.info(f"Processed dataset plots generated and saved in {save_path}.")
    except Exception as e:
        logging.info(f"Error generating processed dataset plots. - {e}")

def find_complete_augmentations_with_labels_and_save():
    """
    Finds patient-video combinations in augmented data that have all four augmentation types 
    (xray, contrast, brightness, translation) along with their lesion labels, orders by severity,
    and saves the results as a JSON file.

    The augmented dataset is loaded from a predefined path in CONFIG.
    The JSON file is saved in CONFIG["OUTPUT_PATH"].

    Returns:
    --------
    None
    """
    try:
        # Load dataset path from CONFIG
        processed_folder = os.path.join(CONFIG["OUTPUT_PATH"], "CADICA_Augmented_Images")
        augmented_data_path = os.path.join(processed_folder, "augmented_train.csv")
        output_json_path = os.path.join(CONFIG["OUTPUT_PATH"], "complete_augmentations_with_labels.json")
        
        # Check if the dataset exists
        if not os.path.isfile(augmented_data_path):
            raise FileNotFoundError(f"Augmented dataset not found at {augmented_data_path}.")

        # Load augmented data
        augmented_data = pd.read_csv(augmented_data_path)

        # Dictionary to store patient-video combinations with their lesion labels and augmentation types
        patient_video_dict = defaultdict(lambda: {"LesionLabel": None, "Augmentations": set()})

        # Regex pattern to extract patient, video, and augmentation type from Frame_path
        pattern = re.compile(r'p(\d+)_v(\d+).*_(xray|contrast|brightness|translation)')

        # Populate patient_video_dict with lesion labels and sets of augmentation types
        for _, row in augmented_data.iterrows():
            match = pattern.search(row['Frame_path'])
            if match:
                patient_video = f"p{match.group(1)}_v{match.group(2)}"
                augmentation_type = match.group(3)
                lesion_label = row['LesionLabel']
                
                patient_video_dict[patient_video]["LesionLabel"] = lesion_label
                patient_video_dict[patient_video]["Augmentations"].add(augmentation_type)

        # Define lesion severity order, reversed from least to most severe
        lesion_severity_order = ["p100", "p99", "p90_98", "p70_90", "p50_70", "p20_50", "p0_20", "nolesion"]

        # Filter to find patient-video combinations that have all four augmentation types
        complete_augmentations = {
            pv: {"LesionLabel": details["LesionLabel"], "Augmentations": list(details["Augmentations"])}
            for pv, details in patient_video_dict.items()
            if len(details["Augmentations"]) == 4
        }

        # Sort results by lesion severity
        sorted_augmentations = {
            pv: details for pv, details in sorted(
                complete_augmentations.items(), 
                key=lambda x: lesion_severity_order.index(x[1]["LesionLabel"])
            )
        }

        # Save sorted results to JSON file
        with open(output_json_path, 'w') as json_file:
            json.dump(sorted_augmentations, json_file, indent=4)

        print(f"Complete augmentations with labels saved to {output_json_path}.")
    
    except FileNotFoundError as fnf_error:
        print(f"File error: {fnf_error}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

def plot_augmented_bboxes(patient_video: str, frame_id: str):
    """
    Plot a 1x4 subplot with each augmentation type for a given patient and frame.

    Parameters:
    patient_video (str): The identifier for the patient and video combination (e.g., "p34_v9").
    frame_id (str): The frame identifier (e.g., "00035").
    """
    try:
        # Set up augmentation types
        augmentations = ["Brightness", "Contrast", "Translation", "X_ray artificial noise"]
        
        # Load the filtered data for the original and augmented bounding boxes
        original_path = os.path.join(CONFIG["OUTPUT_PATH"], "CADICA_Holdout_Info", "train.csv")
        train_df = pd.read_csv(original_path)
        augmentations_folder = os.path.join(CONFIG["OUTPUT_PATH"], "CADICA_Augmented_Images")
        augmented_train_df = pd.read_csv(os.path.join(augmentations_folder, "augmented_train.csv"))
        save_path = os.path.join(CONFIG["OUTPUT_PATH"], "PAPER_Figures")
        os.makedirs(save_path, exist_ok=True)
        save_formats = CONFIG["FIGURE_FORMATS"]
        # Filter original bounding boxes for the specified patient and frame
        train_df = train_df[train_df["Frame_path"].str.contains(patient_video) & train_df["Frame_path"].str.contains(frame_id)]
        
        # Filter augmented bounding boxes for specified augmentations
        augmented_train_df = augmented_train_df[augmented_train_df["Frame_path"].str.contains(patient_video) & augmented_train_df["Frame_path"].str.contains(frame_id)]
        
        # Identify each augmentation
        brightness = augmented_train_df[augmented_train_df["Groundtruth_path"].str.contains("brightness_1")].iloc[0, :]
        contrast = augmented_train_df[augmented_train_df["Groundtruth_path"].str.contains("contrast_1")].iloc[0, :]
        translation = augmented_train_df[augmented_train_df["Groundtruth_path"].str.contains("translation_1")].iloc[0, :]
        xray = augmented_train_df[augmented_train_df["Groundtruth_path"].str.contains("xray_noise_1")].iloc[0, :]

        # Read the original frame image
        if not train_df.empty:
            frame_path = train_df["Frame_path"].values[0]
            original_image = cv2.imread(frame_path)
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            print("Original frame image not found for specified patient_video and frame_id.")
            return

        # Extract bounding box file paths for original and augmentations
        original_bbox_path = train_df["Groundtruth_path"].values[0]
        augmentation_data = {
            "Brightness": (brightness["Frame_path"], brightness["Groundtruth_path"]),
            "Contrast": (contrast["Frame_path"], contrast["Groundtruth_path"]),
            "Translation": (translation["Frame_path"], translation["Groundtruth_path"]),
            "X_ray artificial noise": (xray["Frame_path"], xray["Groundtruth_path"])
        }

        # Create 1x4 subplot for the augmentations
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        for i, (aug, (aug_image_path, aug_bbox_path)) in enumerate(augmentation_data.items()):
            # Load the augmented image for this augmentation
            aug_image = cv2.imread(aug_image_path)
            aug_image_rgb = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)
            
            # Plot the augmented image
            axes[i].imshow(aug_image_rgb)
            axes[i].set_title(f"{aug} Augmentation")
            
            # Plot original bounding boxes on augmented image
            with open(original_bbox_path, 'r') as f:
                for line in f:
                    x, y, w, h, _ = line.split()
                    axes[i].add_patch(plt.Rectangle((int(x), int(y)), int(w), int(h), 
                                                    edgecolor='#00B945', facecolor='none', lw=2, label="Original", linestyle="-"))

            # Plot new bounding boxes for the augmentation
            with open(aug_bbox_path, 'r') as f:
                for line in f:
                    x, y, w, h, _ = line.split()
                    axes[i].add_patch(plt.Rectangle((int(x), int(y)), int(w), int(h), 
                                                    edgecolor='#FF9500', facecolor='none', lw=2, label="Augmented"))

            # Remove axis for clarity
            axes[i].axis("off")

        # Display legend for bounding boxes
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, ["Original Bounding Box", "Augmented Bounding Box"], loc="lower center", ncol=2)

        plt.show()
        # Save figures in specified formats
        for fmt in save_formats:
            fig.savefig(os.path.join(save_path, f'augmented_examples.{fmt}'), format=fmt)

        plt.show()
        logging.info(f"Processed dataset plots generated and saved in {save_path}.")
    except Exception as e:
        logging.info(f"Error generating processed dataset plots. - {e}")


def main():
    #original_dataset_visualization()
    processed_dataset_visualization()

    #find_complete_augmentations_with_labels_and_save()
    # Run the function with the provided patient_video
    # Test the function with example values
    plot_augmented_bboxes("p34_v9", "00045")
if __name__ == "__main__":
    main()