#!/usr/bin/env python3

# Ideas for visualization in the dataset:
# 1. Samples per set (training/validation/test). Include a subplot with 2 plots: 
#       I. Set-based visualization: barplot; X-axis: "training" "validation" "test". Y-axis: counts; bars: "Lesion" "No Lesion" "Total" ✅ 
#       II. Label-based visualization: barplot; X-axis: Diagnostic labels, Y-axis: counts; bars: "Training" "Validation" "Test" ✅ 
# 2. Undersampling/Augmentation types used.
#       I. 4 images from a specific patient that show each augmentation image generated ✅ 
#       II. Augmented images per label: Comparison between original, undersampled, and augmented ✅ 
# 3. Simulated Annealing visualization: One scatterplot per adjusted hyperparameter, mark the best result. X-Axis: hyperparameter values; Y-Axis: fitness ✅ 
# 4. Goal-oriented sensibility study of hyperparameters. 
#       I. lineplot for variation of mAP@50-95 per epoch for every hyperparameter value tried. X-axis: epoch, Y-axis: mAP@50-95, legend: Hyperparameter values ✅ 
#       II. lineplot for variation of mAP@50-95 per hyperparameter value. X-axis: hyperparameter value, Y-axis: mAP@50-95 ✅ 
# 5. Results comparison: Lineplot of mAP@50-95 performance per model. X-axis: epoch, Y-axis: mAP@50-95, one data series per model (baseline, Simulated Annealing, iteration 1, iteration 2) ✅ 

from typing import List, Dict
import pandas as pd
import os
import logging
import json
from collections import defaultdict
import re
from natsort import natsorted
import yaml
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'ieee', 'std-colors'])
plt.rcParams['font.size'] = 12
plt.rcParams.update({'figure.dpi': '300'})
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

CONFIG_PATH = "./scripts/config.yaml"
FIGSIZE = (15,7.5)
SHOW = False

# Set up logging
os.makedirs('./logs', exist_ok=True)
logging.basicConfig(filename='./logs/visualization.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        fig, axs = plt.subplots(1, 2, figsize=FIGSIZE)

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

        if SHOW: plt.show()
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
        SHOW = CONFIG.get("SHOW_PLOTS", False)  # Add this to CONFIG if needed

        # Load original datasets
        original_train = pd.read_csv(os.path.join(original_folder, 'train.csv'))
        original_val = pd.read_csv(os.path.join(original_folder, 'val.csv'))
        original_test = pd.read_csv(os.path.join(original_folder, 'test.csv'))

        # Load processed datasets
        processed_train = pd.read_csv(os.path.join(processed_folder, 'processed_train.csv'))
        processed_val = pd.read_csv(os.path.join(processed_folder, 'processed_val.csv'))
        processed_test = pd.read_csv(os.path.join(processed_test_folder, 'processed_test.csv'))

        # Helper function to parse 'LesionLabel' into lists
        def parse_labels(label_str):
            if pd.isna(label_str) or label_str == 'nolesion':
                return ['nolesion']
            else:
                return label_str.split(',')

        # Parse 'LesionLabel' into lists
        for df in [original_train, original_val, original_test, processed_train, processed_val, processed_test]:
            df['LesionLabelList'] = df['LesionLabel'].apply(parse_labels)

        # Helper function to categorize lesion and non-lesion counts for each set
        def categorize_counts(df, set_name):
            lesion_count = len(df[df['LesionLabelList'].apply(lambda x: 'nolesion' not in x)])
            no_lesion_count = len(df[df['LesionLabelList'].apply(lambda x: x == ['nolesion'])])
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

        # Define label mapping
        label_mapping = {
            "nolesion": "No Lesion",
            "p0_20": r"$< 20\%$",
            "p20_50": r"$[20, 50)\%$",
            "p50_70": r"$[50, 70)\%$",
            "p70_90": r"$[70, 90)\%$",
            "p90_98": r"$[90, 98)\%$",
            "p99": r"$99\%$",
            "p100": r"$100\%$"
        }

        # Helper function to count labels
        from collections import Counter

        def count_labels(df, label_order, set_name):
            label_counts = Counter()
            for labels in df['LesionLabelList']:
                label_counts.update(labels)
            counts = [label_counts.get(label, 0) for label in label_order]
            return pd.DataFrame({set_name: counts}, index=label_order)

        # Calculate original and processed label-based counts
        original_labels = pd.concat([
            count_labels(original_train, label_order, 'Training'),
            count_labels(original_val, label_order, 'Validation'),
            count_labels(original_test, label_order, 'Test')
        ], axis=1)
        # Apply label mapping to index
        original_labels.index = original_labels.index.map(label_mapping)

        processed_labels = pd.concat([
            count_labels(processed_train, label_order, 'Training'),
            count_labels(processed_val, label_order, 'Validation'),
            count_labels(processed_test, label_order, 'Test')
        ], axis=1)
        # Apply label mapping to index
        processed_labels.index = processed_labels.index.map(label_mapping)

        # Determine max y-axis limit
        max_y = max(processed_counts[['Lesion', 'No Lesion', 'Total']].values.max(), processed_labels.values.max())

        # Plotting with adjusted y-axis
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))  # Adjust figsize as needed

        # Plot 1: Processed set-based visualization with markers for original values
        offset = 0.165
        processed_counts.plot(kind='bar', x='Set', y=['Lesion', 'No Lesion', 'Total'], ax=axs[0], color=['blue', 'green', 'purple'])
        for i, row in original_counts.iterrows():
            axs[0].plot([i - offset, i, i + offset], row[['Lesion', 'No Lesion', 'Total']], "^", color="red", markersize=7, 
                        label="Original Count" if i == 0 else "")
        axs[0].set_title("Processed Sample Counts per Set")
        axs[0].set_ylabel("Counts")
        axs[0].set_ylim(0, max_y * 1.1)  # Slightly increase y limit
        axs[0].legend()
        axs[0].tick_params(axis='x', rotation=0)  # Rotate x-axis labels

        # Plot 2: Processed label-based visualization with markers for original values
        processed_labels.plot(kind='bar', ax=axs[1])
        for i, label in enumerate(processed_labels.index):
            axs[1].plot([i - offset, i, i + offset], original_labels.loc[label, ['Training', 'Validation', 'Test']], "x", color="red", markersize=7, 
                        label="Original Count" if i == 0 else "")
        axs[1].set_title("Processed Label-based Sample Counts")
        axs[1].set_ylim(0, max_y * 1.1)
        axs[1].set_xlabel("Diagnostic lesion label")
        axs[1].legend()
        axs[1].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability

        for ax in axs:
            ax.spines[['right', 'top']].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
        plt.tight_layout()

        # Save figures in specified formats
        for fmt in save_formats:
            fig.savefig(os.path.join(save_path, f'processed_dataset_visualizations.{fmt}'), format=fmt)

        if SHOW: plt.show()
        logging.info(f"Processed dataset plots generated and saved in {save_path}.")
    except Exception as e:
        logging.error(f"Error generating processed dataset plots. - {e}")

def plot_hyperparameter_results(
    sampler_csv_paths: Dict[str, Dict[str, str]], 
    output_path: str, 
    output_format: str = "png",
    metric_column: Dict[str, str] = {"F1-Score": "objective_0"},
    whiskers: bool = False
):
    """
    Generate a single visualization of a chosen metric per trial for hyperparameter optimization samplers.
    Adds whiskers to optimal trials based on F1-Score per epoch computed from results.csv files.

    Parameters:
        sampler_csv_paths (Dict[str, Dict[str, str]]): Dictionary with sampler names as keys and dictionaries containing:
            - "path": Path to the CSV file.
            - "color": Color to plot for the sampler.
            - "results_root_folder": Path to the root folder containing results for optimal trials.
        output_path (str): Path to save the output figure.
        output_format (str): Format to save the figure (e.g., 'png', 'svg', 'pdf').
        metric_column (Dict[str, str]): Dictionary with display name as key and column name as value to plot on Y-axis.
    """
    # Extract metric name and column
    metric_name, metric_col = list(metric_column.items())[0]

    # Helper function to calculate the optimal front
    def calculate_optimal_front(trials: List[int], scores: List[float]):
        frontier_x, frontier_y = [], []
        best_so_far = -float("inf")
        for trial, score in zip(trials, scores):
            if score > best_so_far:
                if frontier_x:
                    frontier_x.append(trial)
                    frontier_y.append(best_so_far)  # Horizontal line
                best_so_far = score
                frontier_x.append(trial)
                frontier_y.append(score)  # Vertical line
        if trials[-1] != frontier_x[-1]:
            frontier_x.append(trials[-1])  # Extend horizontally to the last trial
            frontier_y.append(best_so_far)
        return frontier_x, frontier_y

    # Helper function to calculate F1-Score for a trial
    def compute_f1_score(results_file: str):
        df = pd.read_csv(results_file)
        precision = df["metrics/precision(B)"]
        recall = df["metrics/recall(B)"]
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1_score

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    legends = []

    # Plot each sampler
    for sampler_name, sampler_data in sampler_csv_paths.items():
        df = pd.read_csv(sampler_data["path"])
        trials = df.index.values
        scores = df[metric_col]
        
        # Calculate optimal front
        frontier_x, frontier_y = calculate_optimal_front(trials, scores)
        
        # Plot all trials
        ax.scatter(trials, scores, s=10, color=sampler_data["color"], alpha=0.5, label=sampler_name)
        
        # Plot optimal frontier
        ax.step(frontier_x, frontier_y, where='post', color=sampler_data["color"], linewidth=2)
        
        # Add whiskers for optimal trials
        if whiskers:
            optimal_trials = [trial for trial, _ in zip(frontier_x[::2], frontier_y[::2])]
            for trial, mean_f1 in zip(optimal_trials, frontier_y[::2]):
                if sampler_name != "Simulated Annealing":
                    results_file = os.path.join(sampler_data["results_root_folder"], f"trial_{trial}_training", "results.csv")
                else:
                    results_file = os.path.join(sampler_data["results_root_folder"], f"simulated_annealing{trial}", "results.csv")
                if os.path.exists(results_file):
                    f1_scores = compute_f1_score(results_file)
                    min_f1, max_f1 = f1_scores.min(), f1_scores.max()
                    lower_err = max(0, mean_f1 - min_f1)
                    upper_err = max(0, max_f1 - mean_f1)
                    ax.errorbar(trial, mean_f1,
                                yerr=[[lower_err], [upper_err]],
                                fmt="o", color=sampler_data["color"], capsize=3, alpha=0.7)

        # Update legends
        best_idx = df[metric_col].idxmax()
        best_trial = df.loc[best_idx, "number"]
        best_score = df.loc[best_idx, metric_col]
        legends.append((plt.Line2D([], [], color=sampler_data["color"], linewidth=2),
                        f"{sampler_name} (Best: {best_score:.3f}, Trial: {int(best_trial)})"))

    # Add labels and titles
    ax.set_xlabel("Trial")
    ax.set_ylabel(metric_name)

    for ax in [ax]:
        ax.spines[['right', 'top']].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    # Unified legend
    handles, labels = zip(*legends)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    # Save and show the figure
    plt.tight_layout()
    plt.savefig(f"{output_path}.{output_format}", format=output_format, bbox_inches='tight')
    if SHOW: plt.show()

def plot_hyperparameter_scatter(
    sampler_csv_paths: Dict[str, Dict[str, str]],
    output_path: str,
    output_format: str = "png",
    metric_column: Dict[str, str] = {"F1-Score": "user_attrs_f1_score"}
):
    """
    Generate scatterplots for hyperparameters with performance metric as Y-axis, highlighting the best hyperparameter values
    for each sampler.

    Parameters:
        sampler_csv_paths (Dict[str, Dict[str, str]]): Dictionary with sampler names as keys and dictionaries containing:
            - "path": Path to the CSV file.
            - "color": Color to plot for the sampler.
        output_path (str): Path to save the output figure.
        output_format (str): Format to save the figure (e.g., 'png', 'svg', 'pdf').
        metric_column (Dict[str, str]): Dictionary with display name as key and column name as value for the metric.
    """
    # Extract metric name and column
    metric_name, metric_col = list(metric_column.items())[0]

    # List of hyperparameters to plot
    hyperparameters = [
        "params_batch", "params_box", "params_cls", "params_dfl", "params_lr0",
        "params_lrf", "params_momentum", "params_optimizer", "params_warmup_epochs",
        "params_warmup_momentum", "params_weight_decay", "user_attrs_last_epoch"
    ]

    # Prepare the figure with 2 rows x 6 columns subplots
    fig, axes = plt.subplots(2, 6, figsize=(18, 8), sharey=True)
    axes = axes.ravel()  # Flatten axes for easy indexing

    # Process each hyperparameter
    for idx, hyperparameter in enumerate(hyperparameters):
        ax = axes[idx]
        
        # Process each sampler and plot data
        for sampler_name, sampler_data in sampler_csv_paths.items():
            # Load data
            df = pd.read_csv(sampler_data["path"])
            if hyperparameter not in df.columns or metric_col not in df.columns:
                continue
            
            # Encode categorical variables (e.g., params_optimizer)
            if hyperparameter == "params_optimizer":
                label_encoder = LabelEncoder()
                df[hyperparameter] = label_encoder.fit_transform(df[hyperparameter].astype(str))
            
            # Identify the best point for the given sampler based on the metric
            best_idx = df[metric_col].idxmax()
            best_value = df.loc[best_idx, hyperparameter]
            best_score = df.loc[best_idx, metric_col]
            
            # Scatter plot with all points and highlight the best
            scatter = ax.scatter(
                df[hyperparameter], df[metric_col], color=sampler_data["color"], alpha=0.2, s=20, label=sampler_name if idx == 0 else ""
            )
            ax.scatter(
                best_value, best_score, color=sampler_data["color"], alpha=1.0, edgecolor='black', s=80
            )
            ax.plot([], [], 'o', color=sampler_data["color"], label=sampler_name)  # Dummy plot for legend
        
        # Add titles and labels
        # ax.set_title(hyperparameter.replace("_", " ").title())
        ax.set_xlabel(hyperparameter.replace("_", " ").title())
        if idx % 6 == 0:
            ax.set_ylabel(metric_name)

    for ax in axes:
        ax.spines[['right', 'top']].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    # Unified legend
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Remove duplicate legend entries
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(sampler_csv_paths))

    # Adjust layout
    plt.tight_layout()
    #plt.subplots_adjust(bottom=0.1)

    # Save the figure
    plt.savefig(f"{output_path}.{output_format}", format=output_format, bbox_inches='tight')
    if SHOW: plt.show()

def plot_training_comparison(
    sampler_csv_paths: Dict[str, Dict[str, str]],
    output_path: str,
    output_format: str = "png"
):
    """
    Compare the training performance of baseline and optimized YOLO models by computing F1-Score per epoch,
    alongside train/val box loss and dfl loss plots.

    Parameters:
        sampler_csv_paths (Dict[str, Dict[str, str]]): Dictionary with sampler names as keys and dictionaries containing:
            - "path": Path to the CSV file for training results.
            - "color": Color for the data series.
        output_path (str): Path to save the output figure.
        output_format (str): Format to save the figure (e.g., 'png', 'svg', 'pdf').
    """
    # Initialize the figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    f1_ax, box_loss_ax, dfl_loss_ax = axes

    # Plot each sampler's training data, including Baseline
    for sampler_name, sampler_data in sampler_csv_paths.items():
        df = pd.read_csv(sampler_data["best_trial"])
        if "epoch" not in df.columns or "metrics/precision(B)" not in df.columns or "metrics/recall(B)" not in df.columns:
            continue
        
        # Compute F1-Score per epoch
        precision = df["metrics/precision(B)"]
        recall = df["metrics/recall(B)"]
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        smoothed_f1 = gaussian_filter1d(f1_score, sigma=1.0)
        
        # Subplot 1: F1-Score
        f1_ax.plot(df["epoch"], f1_score, color=sampler_data["color"], linewidth=1, alpha=0.5, label="")
        f1_ax.plot(df["epoch"], smoothed_f1, color=sampler_data["color"], linewidth=2, alpha=1.0, label=f"{sampler_name}")

        # Subplot 2: train/box_loss vs val/box_loss
        if "train/box_loss" in df.columns and "val/box_loss" in df.columns:
            box_loss_ax.plot(df["epoch"], df["train/box_loss"], color=sampler_data["color"], linestyle="--", linewidth=2, label=f"Train Loss (linestyle)")
            box_loss_ax.plot(df["epoch"], df["val/box_loss"], color=sampler_data["color"], linestyle=":", linewidth=2, label=f"Val Loss (linestyle)")
        
        # Subplot 3: train/dfl_loss vs val/dfl_loss
        if "train/dfl_loss" in df.columns and "val/dfl_loss" in df.columns:
            dfl_loss_ax.plot(df["epoch"], df["train/dfl_loss"], color=sampler_data["color"], linestyle="--", linewidth=2, label=f"Train Loss (linestyle)")
            dfl_loss_ax.plot(df["epoch"], df["val/dfl_loss"], color=sampler_data["color"], linestyle=":", linewidth=2, label=f"Val Loss (linestyle)")

    # Subplot 1: F1-Score formatting
    f1_ax.set_title("Performance Comparison")
    f1_ax.set_xlabel("Epoch")
    f1_ax.set_ylabel("F1-Score")
    
    # Subplot 2: Box Loss formatting
    box_loss_ax.set_title("Box Loss")
    box_loss_ax.set_xlabel("Epoch")
    box_loss_ax.set_ylabel("Loss")
    
    # Subplot 3: DFL Loss formatting
    dfl_loss_ax.set_title("DFL Loss")
    dfl_loss_ax.set_xlabel("Epoch")
    dfl_loss_ax.set_ylabel("Loss")
    
    idx = 0
    for ax in axes:
        ax.spines[['right', 'top']].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        
        if idx == 0: text = "a."
        elif idx == 1: text = "b."
        else: text = "c."
        
        ax.text(-0.05, 1.05, text, transform=ax.transAxes, fontsize=12, fontweight='bold',
                va='top', ha='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        idx += 1

    # Unified legend (remove duplicates)
    handles, labels = [], []
    for ax in [f1_ax, box_loss_ax, dfl_loss_ax]:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # Save the figure
    plt.savefig(f"{output_path}.{output_format}", format=output_format, bbox_inches='tight')
    if SHOW: plt.show()

def main():
    # processed_dataset_visualization()
        
    colors = {
        "RANDOM": "#994455",
        "TPE": "#6699CC",
        "GPSAMPLER": "#997700",
        "QMCSAMPLER": "#EE99AA",
        "SIMULATED_ANNEALING": "#004488",
        "BASELINE": "#000000",
    }

    save_plots_path = "/home/mario/Python/Results/Coronariografias/patient_based_non_augmentation/PLOTS"

    base_path = "/home/mario/Python/Results/Coronariografias/patient_based_non_augmentation"
    base_name = "hyperparameter_optimization_results.csv"
    
    # Example Usage
    sampler_paths = {
        "Random Search": {
            "path": os.path.join(base_path, "RANDOM", base_name),
            "color": colors.get("RANDOM"), 
            "results_root_folder": os.path.join(base_path, "RANDOM", "detect"),
            "best_trial": os.path.join(base_path, "RANDOM", "detect", "trial_40_training", "results.csv")
        },
        "Tree-structured Parzen Estimator": {
            "path": os.path.join(base_path, "TPE", base_name),
            "color": colors.get("TPE"), 
            "results_root_folder": os.path.join(base_path, "TPE", "detect"),
            "best_trial": os.path.join(base_path, "TPE", "detect", "trial_139_training", "results.csv")
        },
        "Gaussian Process-Based Algorithm": {
            "path": os.path.join(base_path, "GPSAMPLER", base_name),
            "color": colors.get("GPSAMPLER"),
            "results_root_folder": os.path.join(base_path, "GPSAMPLER", "detect"),
            "best_trial": os.path.join(base_path, "GPSAMPLER", "detect", "trial_121_training", "results.csv")
        },
        #"Quasi Monte Carlo": {
        #    "path": os.path.join(base_path, "QMCSAMPLER", base_name),
        #    "color": colors.get("QMCSAMPLER"), 
        #    "results_root_folder": os.path.join(base_path, "QMCSAMPLER", "detect"),
        #    "best_trial": os.path.join(base_path, "QMCSAMPLER", "detect", "trial_103_training", "results.csv")
        #},
        "Simulated Annealing": {
            "path": os.path.join(base_path, "SIMULATED_ANNEALING", base_name),
            "color": colors.get("SIMULATED_ANNEALING"),
            "results_root_folder": os.path.join(base_path, "SIMULATED_ANNEALING", "detect"),
            "best_trial": os.path.join(base_path, "SIMULATED_ANNEALING", "detect", "simulated_annealing34", "results.csv")
        },
    }




    
    print("Plotting results per trial")
    plot_hyperparameter_results(sampler_paths, 
                                output_path=os.path.join(save_plots_path, "performance_per_trial"), 
                                output_format="svg", 
                                metric_column={"F1-Score": "user_attrs_f1_score"})

    print("Plotting hyperparameter distributions")
    plot_hyperparameter_scatter(sampler_paths, 
                                output_path=os.path.join(save_plots_path, "hyperparameter_scatterplots"), 
                                output_format="svg", 
                                metric_column={"F1-Score": "user_attrs_f1_score"})
    
    print("Plotting training comparison")
    
    baseline_entry = {
        "Baseline": {
            "best_trial": os.path.join(base_path, "BASELINE", "results.csv"),
            "color": colors.get("BASELINE"),
        }
    }   

    
    plot_training_comparison(
        sampler_csv_paths={
            **sampler_paths,  # Existing entries
            **baseline_entry  # Add the new BASELINE entry
        },
        output_path=os.path.join(save_plots_path, "training_comparison"),
        output_format="svg",
    )


if __name__ == "__main__":
    main()