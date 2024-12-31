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

from typing import List, Dict, Any
import pandas as pd
import os
import logging
import yaml
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

def original_dataset_visualization(save_path:str):
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
        
        if not save_path:
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

def processed_dataset_visualization(save_path:str):
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

        if not save_path:
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

#########################################
# RESULTS PER TRIAL, DIFFERENT VERSIONS #
#########################################

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

def plot_hyperparameter_results_with_gaussian(
    sampler_csv_paths: Dict[str, Dict[str, str]], 
    output_path: str, 
    output_format: str = "png",
    metric_column: Dict[str, str] = {"F1-Score": "objective_0"},
    whiskers: bool = False,
    sigma: float = 2.0
):
    """
    Generate a single visualization of a chosen metric per trial for hyperparameter optimization samplers,
    adding a Gaussian-filtered trend line for each sampler.

    This function retains the original logic that:
      - Plots all trials as scatter points.
      - Draws the optimal frontier steps.
      - Optionally adds whiskers to optimal trials.
      - Adds a smoothed (Gaussian-filtered) line to indicate the performance tendency.
    
    Parameters:
        sampler_csv_paths (Dict[str, Dict[str, str]]): Dictionary with sampler names as keys and dictionaries containing:
            - "path": Path to the CSV file.
            - "color": Color to plot for the sampler.
            - "results_root_folder": (Optional) Path to root folder containing results for optimal trials.
        output_path (str): Path to save the output figure.
        output_format (str): Format to save the figure (e.g., 'png', 'svg', 'pdf').
        metric_column (Dict[str, str]): Dictionary with display name as key and column name as value to plot on Y-axis.
        whiskers (bool): Whether to add whiskers for optimal trials.
        sigma (float): Standard deviation for the Gaussian filter used in smoothing the scatter points.
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

    fig, ax = plt.subplots(figsize=(12, 7))
    legends = []

    for sampler_name, sampler_data in sampler_csv_paths.items():
        df = pd.read_csv(sampler_data["path"])
        trials = df.index.values
        scores = df[metric_col].values
        
        # Calculate optimal front
        frontier_x, frontier_y = calculate_optimal_front(trials, scores)

        # Scatter plot for all trials
        ax.scatter(trials, scores, s=10, color=sampler_data["color"], alpha=0.5, label=sampler_name)

        # Plot the optimal frontier
        ax.step(frontier_x, frontier_y, where='post', color=sampler_data["color"], linewidth=2)

        # Gaussian smoothing for the scatter points
        smoothed_scores = gaussian_filter1d(scores, sigma=sigma)
        ax.plot(trials, smoothed_scores, color=sampler_data["color"], linestyle='--', linewidth=2, alpha=0.8)

        # Optionally plot whiskers for optimal trials
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

        # Legend info
        best_idx = df[metric_col].idxmax()
        best_trial = df.loc[best_idx, "number"]
        best_score = df.loc[best_idx, metric_col]
        legends.append((plt.Line2D([], [], color=sampler_data["color"], linewidth=2),
                        f"{sampler_name} (Best: {best_score:.3f}, Trial: {int(best_trial)})"))

    ax.set_xlabel("Trial")
    ax.set_ylabel(metric_name)
    ax.spines[['right', 'top']].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Unified legend
    handles, labels = zip(*legends)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    plt.tight_layout()
    plt.savefig(f"{output_path}.{output_format}", format=output_format, bbox_inches='tight')
    if SHOW: plt.show()

def plot_hyperparameter_results_with_regression(
    sampler_csv_paths: Dict[str, Dict[str, str]], 
    output_path: str, 
    output_format: str = "png",
    metric_column: Dict[str, str] = {"F1-Score": "objective_0"},
    whiskers: bool = False,
    debug: bool = True
):
    """
    Generate a figure with two subplots:
      - Subplot[0]: The original hyperparameter results (scatter + optimal frontier + optional whiskers).
      - Subplot[1]: Scatter + fitted linear regression line for each model.
    
    A single unified legend is created at the end, including:
      {model_name} best trial {best_trial} best value {best_score:.4f} |
      Regression: y={slope:.4f}x+{intercept:.4f} R^2={r2:.4f}

    Parameters:
        sampler_csv_paths (Dict[str, Dict[str, str]]): 
            Dictionary with sampler names as keys and dictionaries containing:
              - "path": Path to the CSV file.
              - "color": Color to plot for the sampler.
              - "results_root_folder": Path to the root folder for optimal trials.
        output_path (str): Path to save the output figure.
        output_format (str): Format to save the figure ('png', 'svg', 'pdf', etc.).
        metric_column (Dict[str, str]): Display name as key and column name as value for the Y-axis.
        whiskers (bool): Whether to add whiskers for optimal trials in subplot 0.
        debug (bool): Whether to print debug info and save CSVs for diagnosing issues.
    """

    metric_name, metric_col = list(metric_column.items())[0]

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
            frontier_x.append(trials[-1])
            frontier_y.append(best_so_far)
        return frontier_x, frontier_y

    def compute_f1_score(results_file: str):
        df = pd.read_csv(results_file)
        precision = df["metrics/precision(B)"]
        recall = df["metrics/recall(B)"]
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1_score

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6))

    # Dictionary to store sampler info for the unified legend
    # Each entry will look like:
    # sampler_info[sampler_name] = {
    #     "color": ...,
    #     "best_trial": ...,
    #     "best_score": ...,
    #     "slope": ...,
    #     "intercept": ...,
    #     "r2": ...
    # }
    sampler_info = {}

    # -------------------------
    # Subplot[0]: Original logic
    # -------------------------
    for sampler_name, sampler_data in sampler_csv_paths.items():
        df = pd.read_csv(sampler_data["path"])

        # Debug: Save/inspect the data
        if debug:
            debug_csv_path = os.path.join(
                os.path.dirname(output_path), 
                f"{sampler_name}_debug_original.csv"
            )
            df.to_csv(debug_csv_path, index=False)
            print(f"[DEBUG] Saved original data for '{sampler_name}' to: {debug_csv_path}")

        trials = df.index.values
        scores = df[metric_col].values

        # Calculate optimal frontier
        frontier_x, frontier_y = calculate_optimal_front(trials, scores)

        # Scatter plot
        ax0.scatter(
            trials, scores, 
            s=10, 
            color=sampler_data["color"], 
            alpha=0.5, 
            label="_nolegend_"  # no legend for subplot 0
        )
        # Plot the frontier
        ax0.step(frontier_x, frontier_y, where='post',
                 color=sampler_data["color"], linewidth=2, label="_nolegend_")

        # Whiskers if requested
        if whiskers:
            optimal_trials = [trial for trial, _ in zip(frontier_x[::2], frontier_y[::2])]
            for trial, mean_f1 in zip(optimal_trials, frontier_y[::2]):
                results_file = None
                if sampler_name != "Simulated Annealing":
                    results_file = os.path.join(
                        sampler_data["results_root_folder"], 
                        f"trial_{trial}_training", 
                        "results.csv"
                    )
                else:
                    results_file = os.path.join(
                        sampler_data["results_root_folder"], 
                        f"simulated_annealing{trial}", 
                        "results.csv"
                    )
                if os.path.exists(results_file):
                    f1_scores = compute_f1_score(results_file)
                    min_f1, max_f1 = f1_scores.min(), f1_scores.max()
                    lower_err = max(0, mean_f1 - min_f1)
                    upper_err = max(0, max_f1 - mean_f1)
                    ax0.errorbar(
                        trial, mean_f1,
                        yerr=[[lower_err], [upper_err]],
                        fmt="o", color=sampler_data["color"], capsize=3, alpha=0.7,
                        label="_nolegend_"
                    )

        # Identify best trial info
        best_idx = df[metric_col].idxmax()
        best_trial = df.loc[best_idx, "number"]
        best_score = df.loc[best_idx, metric_col]

        # Initialize sampler info so we can fill in the regression data later
        sampler_info[sampler_name] = {
            "color": sampler_data["color"],
            "best_trial": int(best_trial),
            "best_score": float(best_score),
            "slope": None,
            "intercept": None,
            "r2": None
        }

    ax0.set_xlabel("Trial")
    ax0.set_ylabel(metric_name)
    ax0.set_title("Original Hyperparameter Results")
    ax0.spines[['right', 'top']].set_visible(False)
    ax0.get_xaxis().tick_bottom()
    ax0.get_yaxis().tick_left()

    # -------------------------
    # Subplot[1]: Linear Regression
    # -------------------------
    for sampler_name, sampler_data in sampler_csv_paths.items():
        df = pd.read_csv(sampler_data["path"])

        # Debug: Save/inspect the data
        if debug:
            debug_csv_path = os.path.join(
                os.path.dirname(output_path), 
                f"{sampler_name}_debug_regression.csv"
            )
            df.to_csv(debug_csv_path, index=False)
            print(f"[DEBUG] Saved regression data for '{sampler_name}' to: {debug_csv_path}")

        # Drop or fill NaNs in the metric column
        if df[metric_col].isnull().any():
            nan_rows = df[df[metric_col].isnull()]
            print(f"[DEBUG] Detected NaN in '{metric_col}' for sampler '{sampler_name}':")
            print(nan_rows)
            df = df.dropna(subset=[metric_col])
            print(f"[DEBUG] Dropped NaN rows for sampler '{sampler_name}'.")

        # If the dataframe is empty after dropping, skip
        if df.empty:
            print(f"[ERROR] No valid rows left for sampler '{sampler_name}' after dropping NaNs.")
            continue

        # Prepare data for regression
        trials = df.index.values.reshape(-1, 1)
        scores = df[metric_col].values

        # Fit linear regression
        try:
            reg = LinearRegression().fit(trials, scores)
            y_pred = reg.predict(trials)
            r2 = r2_score(scores, y_pred)
            slope = reg.coef_[0]
            intercept = reg.intercept_

            # Scatter + regression line
            ax1.scatter(trials, scores, s=10, color=sampler_data["color"], alpha=0.5, label="_nolegend_")
            ax1.plot(trials, y_pred, color=sampler_data["color"], linewidth=2, label="_nolegend_")

            # Save the regression data in our sampler info
            if sampler_name in sampler_info:
                sampler_info[sampler_name]["slope"] = float(slope)
                sampler_info[sampler_name]["intercept"] = float(intercept)
                sampler_info[sampler_name]["r2"] = float(r2)
            else:
                # If for some reason the sampler wasn't in the first subplot
                sampler_info[sampler_name] = {
                    "color": sampler_data["color"],
                    "best_trial": None,
                    "best_score": None,
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r2": float(r2)
                }

        except ValueError as e:
            print(f"[ERROR] Could not fit Linear Regression for sampler '{sampler_name}': {e}")
            continue

    ax1.set_xlabel("Trial")
    ax1.set_ylabel(metric_name)
    ax1.set_title("Linear Regression Models")
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()

    # -------------------------
    # Unified Legend
    # -------------------------
    # Create one legend entry per sampler, with a line in the sampler's color
    legend_handles = []
    legend_labels = []

    subplot_labels = ["a.", "b."]
    for idx, ax in enumerate([ax0, ax1]):
        ax.spines[['right', 'top']].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.text(-0.05, 1.05, subplot_labels[idx], transform=ax.transAxes, fontsize=12, fontweight='bold',
                va='top', ha='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    for sampler_name, info in sampler_info.items():
        line = plt.Line2D([], [], color=info["color"], linewidth=2)
        best_trial = info["best_trial"]
        best_score = info["best_score"]
        slope = info["slope"]
        intercept = info["intercept"]
        r2 = info["r2"]

        # Build label with .4f precision
        # e.g. "Random Search best trial 10 best value 0.9234 | Regression: y=0.1234x+0.5678 R^2=0.4321"
        label_parts = [f"{sampler_name}"]
        if best_trial is not None and best_score is not None:
            label_parts.append(
                f"Value: {best_score:.4f} Trial: {best_trial}"
            )
        if slope is not None and intercept is not None and r2 is not None:
            label_parts.append(
                f"Regression: y={slope:.4f}x+{intercept:.4f} with R$^2$={r2:.4f}"
            )

        label = " | ".join(label_parts)
        legend_handles.append(line)
        legend_labels.append(label)

    # Place the legend at the bottom, in one line (adjust as needed)
    fig.legend(
        legend_handles, 
        legend_labels, 
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=1  # or more columns if you have many samplers
    )

    plt.tight_layout()
    # Make sure legend is not cut off
    plt.subplots_adjust(bottom=0.2)

    # Save and show
    plt.savefig(f"{output_path}.{output_format}", format=output_format, bbox_inches="tight")
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
        ax.set_xlabel(hyperparameter.replace("_", " ").title().lstrip("Params").lstrip("User Attrs"))
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
    alongside train/val box loss, dfl loss, and cls loss plots.

    Parameters:
        sampler_csv_paths (Dict[str, Dict[str, str]]): Dictionary with sampler names as keys and dictionaries containing:
            - "path": Path to the CSV file for training results.
            - "color": Color for the data series.
        output_path (str): Path to save the output figure.
        output_format (str): Format to save the figure (e.g., 'png', 'svg', 'pdf').
    """
    # Initialize the figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    f1_ax, box_loss_ax, dfl_loss_ax, cls_loss_ax = axes

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
            box_loss_ax.plot(df["epoch"], df["train/box_loss"], color=sampler_data["color"], linestyle="--", linewidth=2, label=f"Train Loss")
            box_loss_ax.plot(df["epoch"], df["val/box_loss"], color=sampler_data["color"], linestyle=":", linewidth=2, label=f"Val Loss")
        
        # Subplot 3: train/dfl_loss vs val/dfl_loss
        if "train/dfl_loss" in df.columns and "val/dfl_loss" in df.columns:
            dfl_loss_ax.plot(df["epoch"], df["train/dfl_loss"], color=sampler_data["color"], linestyle="--", linewidth=2, label=f"Train Loss")
            dfl_loss_ax.plot(df["epoch"], df["val/dfl_loss"], color=sampler_data["color"], linestyle=":", linewidth=2, label=f"Val Loss")

        # Subplot 4: train/cls_loss vs val/cls_loss
        if "train/cls_loss" in df.columns and "val/cls_loss" in df.columns:
            cls_loss_ax.plot(df["epoch"], df["train/cls_loss"], color=sampler_data["color"], linestyle="--", linewidth=2, label=f"Train Loss")
            cls_loss_ax.plot(df["epoch"], df["val/cls_loss"], color=sampler_data["color"], linestyle=":", linewidth=2, label=f"Val Loss")

    # Subplot 1: F1-Score formatting
    f1_ax.set_title("Performance Comparison")
    f1_ax.set_xlabel("Epoch")
    f1_ax.set_ylabel("F1-Score")
    
    # Subplot 2: Box Loss formatting
    box_loss_ax.set_title("Box Loss")
    box_loss_ax.set_xlabel("Epoch")
    box_loss_ax.set_ylabel("log-Loss")
    box_loss_ax.set_yscale("log")
    
    # Subplot 3: DFL Loss formatting
    dfl_loss_ax.set_title("DFL Loss")
    dfl_loss_ax.set_xlabel("Epoch")
    dfl_loss_ax.set_ylabel("log-Loss")
    dfl_loss_ax.set_yscale("log")

    # Subplot 4: CLS Loss formatting
    cls_loss_ax.set_title("CLS Loss")
    cls_loss_ax.set_xlabel("Epoch")
    cls_loss_ax.set_ylabel("log-Loss")
    cls_loss_ax.set_yscale("log")
    
    # Format subplot labels (e.g. a., b., c., d.)
    subplot_labels = ["a.", "b.", "c.", "d."]
    for idx, ax in enumerate(axes):
        ax.spines[['right', 'top']].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.text(-0.05, 1.05, subplot_labels[idx], transform=ax.transAxes, fontsize=12, fontweight='bold',
                va='top', ha='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # Create a unified legend (remove duplicates)
    handles, labels = [], []
    for ax in axes:
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

def compute_f1_score(precision: float, recall: float) -> float:
    """
    Computes the F1-Score from precision and recall.
    """
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def aggregate_fold_data(root_folder: str) -> pd.DataFrame:
    """
    Aggregates test metrics and label-specific metrics across models and folds.

    Args:
        root_folder: Root directory containing fold subfolders.

    Returns:
        A DataFrame containing aggregated data with these columns:
            ['Model', 'Fold', 'Overall_F1', 'Overall_mAP50', 'Overall_mAP50-95', 
             'Labels'] 
        where 'Labels' is a dictionary:
            { label_name: (F1, mAP50, mAP50-95) }
        Missing labels per fold are stored as np.nan in that label's tuple.
    """
    records = []
    
    for fold_folder in os.listdir(root_folder):
        fold_path = os.path.join(root_folder, fold_folder)
        if not os.path.isdir(fold_path):
            continue
        
        metrics_file = os.path.join(fold_path, f"{fold_folder}_validation_metrics.csv")
        label_metrics_file = os.path.join(fold_path, f"{fold_folder}_label_validation_results.csv")
        if not (os.path.exists(metrics_file) and os.path.exists(label_metrics_file)):
            print(f"[WARNING] Metrics files missing for {fold_path}. Skipping this folder.")
            continue
        
        # Extract model name (e.g., "RANDOM" from "RANDOM_outer_1_inner_1")
        model_name = fold_folder.split("_outer")[0]
        # Extract outer fold number
        parts = fold_folder.split("_")
        fold_number = parts[2] if len(parts) > 2 else "?"

        overall_df = pd.read_csv(metrics_file)
        label_df = pd.read_csv(label_metrics_file)

        # Filter out 'nolesion' and compute F1
        label_df = label_df[label_df['Label'] != 'nolesion'].copy()
        label_df['F1'] = label_df.apply(
            lambda row: compute_f1_score(row['test/precision'], row['test/recall']), axis=1
        )

        # Compute overall F1
        overall_f1 = compute_f1_score(
            overall_df['test/precision'].iloc[0],
            overall_df['test/recall'].iloc[0]
        )
        overall_map50 = overall_df['test/mAP50'].iloc[0]
        overall_map5095 = overall_df['test/mAP50-95'].iloc[0]

        label_dict = {}
        for _, row in label_df.iterrows():
            label_name = row['Label']
            f1_val = row['F1']
            map50_val = row['test/mAP50']
            map5095_val = row['test/mAP50-95']
            label_dict[label_name] = (f1_val, map50_val, map5095_val)
        
        records.append({
            'Model': model_name,
            'Fold': fold_number,
            'Overall_F1': overall_f1,
            'Overall_mAP50': overall_map50,
            'Overall_mAP50-95': overall_map5095,
            'Labels': label_dict
        })

    df = pd.DataFrame(records)
    return df

def generate_boxplot(df: pd.DataFrame, output_path: str, output_format:str="svg"):
    """
    Generates a boxplot for overall and label-specific F1-scores across models,
    using a predefined label and model order. Uses subplots() to allow customizing
    axis spines and ticks.
    
    Args:
        df: DataFrame from aggregate_fold_data().
        output_path: Path to save the boxplot image.
    """
    import matplotlib.patches as mpatches

    # Desired model order
    model_order = ["TPE", "GPSAMPLER", "RANDOM", "SIMULATED_ANNEALING", "BASELINE"]
    # Color mapping
    colors = {
        "RANDOM": "#994455",
        "TPE": "#6699CC",
        "GPSAMPLER": "#997700",
        "QMCSAMPLER": "#EE99AA",  # If you ever have QMCSAMPLER
        "SIMULATED_ANNEALING": "#004488",
        "BASELINE": "#000000",
    }

    # Desired label order (including Overall)
    label_order = ["Overall", "p100", "p99", "p90_98", "p70_90", "p50_70"]
    label_mapping = {
        "p50_70": r"$[50, 70)\%$",
        "p70_90": r"$[70, 90)\%$",
        "p90_98": r"$[90, 98)\%$",
        "p99": r"$99\%$",
        "p100": r"$100\%$"
    }

    # Build an internal structure for plotting: { label -> { model -> [F1 across folds] } }
    f1_scores = { label: {m: [] for m in model_order} for label in label_order }

    for _, row in df.iterrows():
        model = row['Model']
        if model not in model_order:
            continue
        # Overall
        f1_scores["Overall"][model].append(row['Overall_F1'])
        # Label-specific
        labels_dict = row['Labels']
        for label in labels_dict:
            if label in label_order:  # Only gather if label is recognized in our order
                f1_scores[label][model].append(labels_dict[label][0])  # the F1 is index 0

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    positions = []
    data_to_plot = []
    color_patches = []

    offset = 0.15
    xtick_labels = []

    # We'll iterate over label_order and create multiple sub-positions for each model
    for label_idx, label in enumerate(label_order):
        num_models = len(model_order)
        # center position for this label
        mid_position = label_idx
        start_position = mid_position - (offset * (num_models - 1) / 2.0)

        for m_idx, model in enumerate(model_order):
            all_values = f1_scores[label][model]
            positions.append(start_position + m_idx * offset)
            data_to_plot.append(all_values)
            color_patches.append(colors.get(model, "#000000"))

        xtick_labels.append(label)

    bplot = ax.boxplot(
        data_to_plot, 
        positions=positions, 
        patch_artist=True, 
        widths=offset * 0.8,
        medianprops=dict(color="black")
    )

    for patch, c in zip(bplot['boxes'], color_patches):
        patch.set_facecolor(c)

    # X-ticks
    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels([label_mapping.get(lbl, lbl) for lbl in label_order], rotation=0)

    ax.set_title("Cross Validation F1-Score")
    ax.set_xlabel("Labels")
    ax.set_ylabel("F1-Score")

    # Hide top/right spines and set bottom/left ticks
    ax.spines[['right', 'top']].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Draw horizontal grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Create legend
    legend_handles = []
    for model in model_order:
        if model in colors:
            patch = mpatches.Patch(color=colors[model], label=model)
            legend_handles.append(patch)
    # Place legend outside the lower center
    ax.legend(
        handles=legend_handles, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(model_order),
        frameon=False
    )

    fig.tight_layout()
    # Use bbox_inches="tight" so the legend is included
    plt.savefig(f"{output_path}.{output_format}", format=output_format, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Boxplot saved at {output_path}")


def generate_latex_table(df: pd.DataFrame, output_tex_path: str):
    """
    Generates a LaTeX table summarizing mean and std of F1, mAP50, mAP50-95
    for each label (including Overall) and each model, in a specific model order.
    No bold highlights.

    Args:
        df: DataFrame from aggregate_fold_data().
        output_tex_path: Path to the .tex file to write.
    """
    # Desired model order
    model_order = ["TPE", "GPSAMPLER", "RANDOM", "SIMULATED_ANNEALING", "BASELINE"]
    model_mapping = {
        "TPE": "Tree-Structured Parzen Estimator",
        "GPSAMPLER": "Gaussian Process-Based Algorithm",
        "RANDOM": "Random Search",
        "SIMULATED_ANNEALING": "Simulated Annealing",
        "BASELINE": "Baseline"
    }
    # Desired label order, plus mapping
    label_order = ["Overall", "p100", "p99", "p90_98", "p70_90", "p50_70"]
    label_mapping = {
        "p50_70": r"$[50, 70)\%$",
        "p70_90": r"$[70, 90)\%$",
        "p90_98": r"$[90, 98)\%$",
        "p99": r"$99\%$",
        "p100": r"$100\%$"
    }

    # Collect all (F1, mAP50, mAP50-95) values
    # metrics_dict[label][model] = [(f1, map50, map50-95), ... per fold]
    metrics_dict = {
        label: {m: [] for m in model_order} for label in label_order
    }

    for _, row in df.iterrows():
        model = row['Model']
        if model not in model_order:
            continue

        # Overall
        metrics_dict["Overall"][model].append(
            (row['Overall_F1'], row['Overall_mAP50'], row['Overall_mAP50-95'])
        )

        # Labels
        labels_dict = row['Labels']
        for label in labels_dict:
            if label in label_order:
                metrics_dict[label][model].append(labels_dict[label])

    # Build the final rows
    # For each label and model, compute mean ± std
    table_rows = {label: [] for label in label_order}

    for label in label_order:
        for model in model_order:
            data_list = metrics_dict[label][model]
            if len(data_list) == 0:
                # No data
                f1_str = "N/A"
                map50_str = "N/A"
                map5095_str = "N/A"
            else:
                arr = np.array(data_list, dtype=np.float32)
                mean_vals = np.nanmean(arr, axis=0)
                std_vals = np.nanstd(arr, axis=0)
                mean_f1, mean_mAP50, mean_mAP5095 = mean_vals
                std_f1, std_mAP50, std_mAP5095 = std_vals

                if np.isnan(mean_f1):
                    f1_str = "N/A"
                else:
                    f1_str = f"{mean_f1:.3f} ± {std_f1:.3f}"

                if np.isnan(mean_mAP50):
                    map50_str = "N/A"
                else:
                    map50_str = f"{mean_mAP50:.3f} ± {std_mAP50:.3f}"

                if np.isnan(mean_mAP5095):
                    map5095_str = "N/A"
                else:
                    map5095_str = f"{mean_mAP5095:.3f} ± {std_mAP5095:.3f}"

            table_rows[label].append({
                "Metric": label,
                "Model": model_mapping.get(model),
                "F1": f1_str,
                "mAP50": map50_str,
                "mAP50-95": map5095_str
            })

    # Generate LaTeX code
    latex_lines = []
    latex_lines.append(r"\begin{table}[h!]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\begin{tabular}{l l c c c}")
    latex_lines.append(r"\hline")
    latex_lines.append(r"Metric & Model & F1-Score & mAP50 & mAP50-95 \\")
    latex_lines.append(r"\hline")

    for label in label_order:
        rows_for_label = table_rows[label]
        # Remove models that truly have no entry if you prefer, 
        # but here we'll keep them all in the specified order.
        # The first row will contain the label, subsequent ones are blank in that column.
        if all((r["F1"] == "N/A" and r["mAP50"] == "N/A" and r["mAP50-95"] == "N/A") for r in rows_for_label):
            # means no data at all for that label
            continue

        # We'll map the label if possible
        label_display = label_mapping.get(label, label)

        first_row = rows_for_label[0]
        latex_lines.append(
            f"{label_display} & {first_row['Model']} & {first_row['F1']} & {first_row['mAP50']} & {first_row['mAP50-95']} \\\\"
        )
        for row in rows_for_label[1:]:
            latex_lines.append(
                f" & {row['Model']} & {row['F1']} & {row['mAP50']} & {row['mAP50-95']} \\\\"
            )
        latex_lines.append(r"\hline")

    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\caption{Performance metrics (mean $\pm$ std) across models and labels.}")
    latex_lines.append(r"\label{tab:performance_metrics}")
    latex_lines.append(r"\end{table}")

    with open(output_tex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"[INFO] LaTeX table saved at {output_tex_path}")

def main():
    colors: Dict[str, str] = {
        "RANDOM": "#994455",
        "TPE": "#6699CC",
        "GPSAMPLER": "#997700",
        "QMCSAMPLER": "#EE99AA",
        "SIMULATED_ANNEALING": "#004488",
        "BASELINE": "#000000",
    }

    save_plots_path: str = "/home/mario/Python/Results/Coronariografias/patient_based_non_augmentation/train_val/PLOTS"
    os.makedirs(save_plots_path, exist_ok=True)
    
    base_path: str = "/home/mario/Python/Results/Coronariografias/patient_based_non_augmentation/train_val"
    base_name: str = "hyperparameter_optimization_results.csv"
    
    image_format: str = "pdf"
    
    processed_dataset_visualization(save_path=save_plots_path)

    
    sampler_paths: Dict[str, Dict[str, Any]] = {
        "Random Search": {
            "path": os.path.join(base_path, "RANDOM", base_name),
            "color": colors.get("RANDOM"), 
            "results_root_folder": os.path.join(base_path, "RANDOM", "detect"),
            "best_trial": os.path.join(base_path, "RANDOM", "detect", "trial_71_training", "results.csv")
        },
        "Tree-structured Parzen Estimator": {
            "path": os.path.join(base_path, "TPE", base_name),
            "color": colors.get("TPE"), 
            "results_root_folder": os.path.join(base_path, "TPE", "detect"),
            "best_trial": os.path.join(base_path, "TPE", "detect", "trial_175_training", "results.csv")
        },
        "Gaussian Process-Based Algorithm": {
            "path": os.path.join(base_path, "GPSAMPLER", base_name),
            "color": colors.get("GPSAMPLER"),
            "results_root_folder": os.path.join(base_path, "GPSAMPLER", "detect"),
            "best_trial": os.path.join(base_path, "GPSAMPLER", "detect", "trial_46_training", "results.csv")
        },
        "Simulated Annealing": {
            "path": os.path.join(base_path, "SIMULATED_ANNEALING", base_name),
            "color": colors.get("SIMULATED_ANNEALING"),
            "results_root_folder": os.path.join(base_path, "SIMULATED_ANNEALING", "detect"),
            "best_trial": os.path.join(base_path, "SIMULATED_ANNEALING", "detect", "simulated_annealing105", "results.csv")
        },
    }

    # Pipeline

    print("Plotting results per trial")
    plot_hyperparameter_results(sampler_paths, 
                                output_path=os.path.join(save_plots_path, "performance_per_trial"), 
                                output_format=image_format, 
                                metric_column={"F1-Score": "value"})
    plot_hyperparameter_results_with_gaussian(sampler_paths, 
                                output_path=os.path.join(save_plots_path, "performance_per_trial_gaussian"), 
                                output_format=image_format, 
                                metric_column={"F1-Score": "value"})
    
    plot_hyperparameter_results_with_regression(sampler_paths, 
                                output_path=os.path.join(save_plots_path, "performance_per_trial_regression"), 
                                output_format=image_format, 
                                metric_column={"F1-Score": "value"},
                                debug=False)

    print("Plotting hyperparameter distributions")
    plot_hyperparameter_scatter(sampler_paths, 
                                output_path=os.path.join(save_plots_path, "hyperparameter_scatterplots"), 
                                output_format=image_format, 
                                metric_column={"F1-Score": "value"})
    
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
        output_format=image_format,
    )
    
    root_folder_cv = os.path.join(base_path, "CV")
    output_boxplot = os.path.join(save_plots_path, "boxplots_cross_validation")
    output_tex = os.path.join(save_plots_path, "performance_table.tex")

    print("Plotting cross-validation performance results")
    aggregated_df = aggregate_fold_data(root_folder_cv)
    generate_boxplot(aggregated_df, output_boxplot, output_format=image_format)
    print("Generating cross-validation performance table")
    generate_latex_table(aggregated_df, output_tex)


if __name__ == "__main__":
    main()