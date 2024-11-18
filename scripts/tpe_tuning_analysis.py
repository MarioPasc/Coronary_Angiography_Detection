"""
Visualizations:
    1. Slice plot. Hyperparameter value in the X-axis, Objective Value in the Y-axis. Trial could be a colormap. 
    2. Hyperparameter importance. It could accompany the visualization 1 in some way. We can measure the hyperparameter importance with
       this optuna.importance submodule: https://optuna.readthedocs.io/en/stable/reference/importance.html 
    3. Epoch evolution of trials against objective value. X-axis is epoch, Y-axis is Objective Value. We can include trials that are 
       completed and pruned or only completed trials. Every trial that is not the best one can be marked grey. 
    4. Trial against Objective Value. X-axis is trial, Y-axis is the Objective Value. We can put this visualization next to plot 3.    
    5. Contour plots of the latent hyperparameter space. We can create a dataframe with each hyperparameter on one column, and add the Objective
       Value obtained for the trial corresponding to that specific se of hyperparameters. We could apply dimensionality reduction to the 
       hyperparameters, keeping the explained varianze high, and plot a contour plot of PC0 vs PC1, where the colour is the mAP@50-95 obtained. 
    6. Pairplot with Contours. We could also see the pariwise distribution of hyperparameters in a pairplot, and add the contour regions based on
       the Objective Value.

"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, Tuple, List, Any
import scienceplots
plt.style.use(['science', 'ieee', 'std-colors'])
plt.rcParams['font.size'] = 12
plt.rcParams.update({'figure.dpi': '300'})
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

PATH = "/home/mariopasc/Python/Results/Coronariografias/patient-based/TPE_Sampler/First_200_trials_YOLOv8"
SHOW = False
FORMAT = ".png"

# Set up logging
logging.basicConfig(filename='logs/visualization_patient_based.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s:%(message)s')


def read_trials_csv(trials_csv_path: str) -> pd.DataFrame:
    """
    Reads the trials CSV file and returns a DataFrame.

    Args:
        trials_csv_path (str): Path to the trials CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the trials data.

    Throws:
        Exception: If the file cannot be read or does not exist.
    """
    try:
        trials_df: pd.DataFrame = pd.read_csv(trials_csv_path)
        return trials_df
    except Exception as e:
        logging.error(f"Error reading trials CSV file: {e}")
        raise


def get_trial_state_mapping(trials_df: pd.DataFrame) -> Dict[int, str]:
    """
    Creates a mapping from trial number to its state.

    Args:
        trials_df (pd.DataFrame): DataFrame containing trials data.

    Returns:
        Dict[int, str]: Dictionary mapping trial number to state.

    Throws:
        Exception: If required columns are missing.
    """
    try:
        trial_number_series: pd.Series = trials_df['number']
        trial_state_series: pd.Series = trials_df['state']
        trial_state: Dict[int, str] = dict(zip(trial_number_series, trial_state_series))
        return trial_state
    except KeyError as e:
        logging.error(f"Required column missing in trials DataFrame: {e}")
        raise
    except Exception as e:
        logging.error(f"Error creating trial state mapping: {e}")
        raise


def process_trial_directories(runs_dir: str) -> Tuple[Dict[int, pd.DataFrame], Dict[int, float]]:
    """
    Processes each trial directory and extracts the trial data.

    Args:
        runs_dir (str): Path to the runs directory.

    Returns:
        Tuple[Dict[int, pd.DataFrame], Dict[int, float]]:
            - trials_data: Mapping of trial number to its results DataFrame.
            - trial_max_map: Mapping of trial number to its maximum mAP@50-95.

    Throws:
        Exception: If errors occur during processing.
    """
    try:
        trials_data: Dict[int, pd.DataFrame] = {}
        trial_max_map: Dict[int, float] = {}

        trial_dirs: List[str] = glob.glob(os.path.join(runs_dir, 'trial_*_training'))

        for trial_dir in trial_dirs:
            trial_name: str = os.path.basename(trial_dir)
            trial_number_str: str = trial_name.split('_')[1]
            try:
                trial_number: int = int(trial_number_str)
            except ValueError as e:
                logging.error(f"Invalid trial number in directory name {trial_name}: {e}")
                continue

            results_csv_path: str = os.path.join(trial_dir, 'results.csv')

            if os.path.exists(results_csv_path):
                try:
                    results_df: pd.DataFrame = pd.read_csv(results_csv_path)

                    if 'epoch' in results_df.columns and 'metrics/mAP50-95(B)' in results_df.columns:
                        data_df: pd.DataFrame = results_df[['epoch', 'metrics/mAP50-95(B)']]
                        trials_data[trial_number] = data_df

                        max_map: float = results_df['metrics/mAP50-95(B)'].max()
                        trial_max_map[trial_number] = max_map
                    else:
                        logging.error(f"Trial {trial_number}: Required columns not found in results.csv")
                except Exception as e:
                    logging.error(f"Error reading results.csv for trial {trial_number}: {e}")
            else:
                logging.error(f"Trial {trial_number}: results.csv not found in {trial_dir}")

        return trials_data, trial_max_map
    except Exception as e:
        logging.error(f"Error processing trial directories: {e}")
        raise


def identify_best_trial(trial_max_map: Dict[int, float]) -> int:
    """
    Identifies the trial with the highest maximum mAP@50-95.

    Args:
        trial_max_map (Dict[int, float]): Mapping of trial number to max mAP@50-95.

    Returns:
        int: Trial number of the best trial.

    Throws:
        Exception: If trial_max_map is empty.
    """
    try:
        if not trial_max_map:
            raise ValueError("No trial data available to identify the best trial.")
        best_trial_number: int = max(trial_max_map, key=trial_max_map.get)
        return best_trial_number
    except Exception as e:
        logging.error(f"Error identifying best trial: {e}")
        raise


def compute_trial_statistics(trials_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Computes the percentages of completed and pruned trials.

    Args:
        trials_df (pd.DataFrame): DataFrame containing trials data.

    Returns:
        Tuple[float, float]: Completed percentage, Pruned percentage.

    Throws:
        Exception: If required columns are missing.
    """
    try:
        total_trials: int = len(trials_df)
        completed_trials_df: pd.DataFrame = trials_df[trials_df['state'] == 'COMPLETE']
        pruned_trials_df: pd.DataFrame = trials_df[trials_df['state'] == 'PRUNED']

        completed_count: int = len(completed_trials_df)
        pruned_count: int = len(pruned_trials_df)

        completed_percentage: float = (completed_count / total_trials) * 100 if total_trials > 0 else 0
        pruned_percentage: float = (pruned_count / total_trials) * 100 if total_trials > 0 else 0

        return completed_percentage, pruned_percentage
    except KeyError as e:
        logging.error(f"Required column missing in trials DataFrame: {e}")
        raise
    except Exception as e:
        logging.error(f"Error computing trial statistics: {e}")
        raise


def plot_trial_data(trials_data: Dict[int, pd.DataFrame],
                    trial_state: Dict[int, str],
                    best_trial_number: int,
                    trial_max_map: Dict[int, float],
                    completed_percentage: float,
                    pruned_percentage: float) -> None:
    """
    Plots mAP@50-95 over epochs for all trials, with the best trial plotted last.

    Args:
        trials_data (Dict[int, pd.DataFrame]): Mapping of trial number to results DataFrame.
        trial_state (Dict[int, str]): Mapping of trial number to state.
        best_trial_number (int): Trial number of the best trial.
        trial_max_map (Dict[int, float]): Mapping of trial number to max mAP@50-95.
        completed_percentage (float): Percentage of completed trials.
        pruned_percentage (float): Percentage of pruned trials.

    Returns:
        None

    Throws:
        Exception: If plotting fails.
    """
    try:
        import os
        import matplotlib.pyplot as plt
        from typing import Any, Dict, List

        plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)

        handles_dict: Dict[str, Any] = {}
        labels_dict: Dict[str, str] = {}

        # Plot all trials except the best trial
        for trial_number, data in trials_data.items():
            if trial_number == best_trial_number:
                continue  # Skip the best trial for now

            state: str = trial_state.get(trial_number, 'UNKNOWN')

            if state == 'COMPLETE':
                color: str = 'blue'
                label: str = f'Completed trials ({completed_percentage:.1f}\%)'
                alpha: float = 0.5
            elif state == 'PRUNED':
                color: str = 'grey'
                label: str = f'Pruned trials ({pruned_percentage:.1f}\%)'
                alpha: float = 0.5
            else:
                color: str = 'black'
                label: str = 'Unknown state'
                alpha: float = 0.5

            line, = ax.plot(data['epoch'], data['metrics/mAP50-95(B)'], color=color, alpha=alpha)

            if label not in labels_dict:
                handles_dict[label] = line
                labels_dict[label] = label

        # Now plot the best trial
        best_trial_data: pd.DataFrame = trials_data[best_trial_number]
        max_map: float = trial_max_map[best_trial_number]
        line_best, = ax.plot(best_trial_data['epoch'], best_trial_data['metrics/mAP50-95(B)'],
                             color='red', alpha=1.0, linewidth=1.7)
        label_best: str = f'Best trial (max mAP@50-95: {max_map:.4f})'

        handles_dict[label_best] = line_best
        labels_dict[label_best] = label_best

        # Update legend handles and labels
        handles: List[Any] = list(handles_dict.values())
        labels: List[str] = list(labels_dict.keys())

        plt.xlabel('Epoch')
        plt.ylabel('mAP@50-95')
        plt.title('mAP@50-95 over Epochs for All Trials')

        # Place legend outside the plot on the lower center
        plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.tight_layout()
        plt.grid(True)
        plt.xlim(0, 100)
        plt.ylim(0, 0.2)


        ax.spines[['right', 'top']].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()


        plt.savefig(os.path.join(PATH, f"evolution_by_epochs{FORMAT}"))
        if SHOW: plt.show()
    except Exception as e:
        import logging
        logging.error(f"Error plotting trial data: {e}")
        raise


def plot_hyperparameter_slice_plots(trials_df: pd.DataFrame,
                                    trial_max_map: Dict[int, float],
                                    trial_state: Dict[int, str],
                                    best_trial_number: int,
                                    baseline_results: Dict[str, Any]) -> None:
    """
    Plots slice plots for each hyperparameter, showing mAP@50-95 vs hyperparameter values.

    Args:
        trials_df (pd.DataFrame): DataFrame containing trials data, including hyperparameters.
        trial_max_map (Dict[int, float]): Mapping of trial number to max mAP@50-95.
        trial_state (Dict[int, str]): Mapping of trial number to state.
        best_trial_number (int): Trial number of the best trial.
        baseline_results (Dict[str, Any]): Baseline hyperparameters and results.
            - Keys are hyperparameter names and their values.
            - Key 'results' contains the path to the baseline's results.csv file.

    Returns:
        None

    Throws:
        Exception: If plotting fails.
    """
    try:
        import os
        import logging
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.lines import Line2D
        from typing import List, Dict, Any

        # Extract hyperparameter columns
        hyperparameter_columns: List[str] = [col for col in trials_df.columns if col.startswith('params_')]

        # Select relevant columns
        trial_params_df: pd.DataFrame = trials_df[['number', 'state'] + hyperparameter_columns]

        # Add max mAP@50-95 to the DataFrame
        trial_params_df['max_mAP'] = trial_params_df['number'].map(trial_max_map)

        # Remove trials without max_mAP
        trial_params_df = trial_params_df.dropna(subset=['max_mAP'])

        # Get baseline hyperparameters and max mAP
        baseline_params: Dict[str, Any] = baseline_results.copy()
        baseline_results_path: str = baseline_params.pop('results', None)
        if baseline_results_path is None:
            raise ValueError("Baseline results path not provided in 'results' key of baseline_results")

        # Function to get baseline max mAP@50-95
        def get_baseline_max_map(baseline_results_path: str) -> float:
            """
            Reads the baseline results.csv file and returns the maximum mAP@50-95.

            Args:
                baseline_results_path (str): Path to the baseline's results.csv file.

            Returns:
                float: Maximum mAP@50-95 value from the baseline's results.csv.

            Throws:
                Exception: If the file cannot be read or the required column is missing.
            """
            try:
                baseline_results_df: pd.DataFrame = pd.read_csv(baseline_results_path)
                if 'metrics/mAP50-95(B)' in baseline_results_df.columns:
                    max_map: float = baseline_results_df['metrics/mAP50-95(B)'].max()
                    return max_map
                else:
                    raise ValueError("Required column 'metrics/mAP50-95(B)' not found in baseline results.csv")
            except Exception as e:
                logging.error(f"Error reading baseline results.csv: {e}")
                raise

        baseline_max_map: float = get_baseline_max_map(baseline_results_path)

        # Create the figure and subplots
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 10, figure=fig)

        axes: List[plt.Axes] = []

        # Top row plots (5 plots)
        axes.append(fig.add_subplot(gs[0, 0:2]))
        axes.append(fig.add_subplot(gs[0, 2:4]))
        axes.append(fig.add_subplot(gs[0, 4:6]))
        axes.append(fig.add_subplot(gs[0, 6:8]))
        axes.append(fig.add_subplot(gs[0, 8:10]))

        # Bottom row plots (4 plots shifted right)
        axes.append(fig.add_subplot(gs[1, 1:3]))
        axes.append(fig.add_subplot(gs[1, 3:5]))
        axes.append(fig.add_subplot(gs[1, 5:7]))
        axes.append(fig.add_subplot(gs[1, 7:9]))


        # Find overall min and max mAP values for consistent Y-axis limits
        all_mAP_values: List[float] = trial_params_df['max_mAP'].tolist()
        all_mAP_values.append(baseline_max_map)
        y_min: float = min(all_mAP_values)
        y_max: float = max(all_mAP_values)
        y_range: float = y_max - y_min
        y_margin: float = y_range * 0.05  # 5% margin
        y_lim: Tuple[float, float] = (y_min - y_margin, y_max + y_margin)

        # Plotting for each hyperparameter
        for idx, (hyperparameter, ax) in enumerate(zip(hyperparameter_columns, axes)):
            # Extract data
            hp_values: pd.Series = trial_params_df[hyperparameter]
            max_mAP_values: pd.Series = trial_params_df['max_mAP']
            trial_numbers: pd.Series = trial_params_df['number']
            states: pd.Series = trial_params_df['state']

            # Masks for trial categories
            best_mask: pd.Series = trial_params_df['number'] == best_trial_number
            completed_mask: pd.Series = (trial_params_df['state'] == 'COMPLETE') & (~best_mask)
            pruned_mask: pd.Series = trial_params_df['state'] == 'PRUNED'

            # Plot pruned trials
            ax.scatter(trial_params_df.loc[pruned_mask, hyperparameter],
                       trial_params_df.loc[pruned_mask, 'max_mAP'],
                       color='grey', marker='o', s=50, alpha=0.5)

            # Plot completed trials
            ax.scatter(trial_params_df.loc[completed_mask, hyperparameter],
                       trial_params_df.loc[completed_mask, 'max_mAP'],
                       color='blue', marker='o', s=50, alpha=0.5)

            # Plot best trial
            if best_mask.any():
                best_trial_mAP: float = trial_params_df.loc[best_mask, 'max_mAP'].values[0]
                best_trial_hp_value: float = trial_params_df.loc[best_mask, hyperparameter].values[0]
                ax.scatter(best_trial_hp_value, best_trial_mAP,
                           color='red', marker='o', s=100, alpha=1.0)
            else:
                logging.error("Best trial data not found in the trials DataFrame.")

            # Plot baseline
            baseline_hp_value: Any = baseline_params.get(hyperparameter)
            if baseline_hp_value is not None:
                ax.scatter(baseline_hp_value, baseline_max_map,
                           color='green', marker='o', s=100, alpha=1.0)
            else:
                logging.warning(f"Baseline value for {hyperparameter} not provided.")

            # Set title with hyperparameter name and best trial value
            hyperparam_name: str = hyperparameter.replace('params_', '')
            best_trial_hp_value_formatted: str = f"{best_trial_hp_value:.5f}"
            ax.set_title(f"{hyperparam_name} (Best: {best_trial_hp_value_formatted})")

            # Set x-axis label without "params_" prefix
            ax.set_xlabel(hyperparam_name)

            # Only keep Y-axis label on the first column
            if idx % 5 == 0:
                ax.set_ylabel('mAP@50-95')
            else:
                ax.set_ylabel('')

            # Set consistent Y-axis limits
            ax.set_ylim(y_lim)
            ax.spines[['right', 'top']].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        # Adjust layout to make room for legend
        plt.subplots_adjust(bottom=0.2)

        # Create custom legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=f'Best trial (max mAP@50-95: {best_trial_mAP:.4f}, Trial: {best_trial_number})',
                   markerfacecolor='red', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Completed trials',
                   markerfacecolor='blue', markersize=7),
            Line2D([0], [0], marker='o', color='w', label='Pruned trials',
                   markerfacecolor='grey', markersize=7),
            Line2D([0], [0], marker='o', color='w', label=f'Baseline (max mAP@50-95: {baseline_max_map:.4f})',
                   markerfacecolor='green', markersize=10)
        ]

        # Add legend to the figure and adjust its position
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0.01, 1, 1])  # Leave space at the bottom for the legend
        plt.savefig(os.path.join(PATH, f"sliceplot{FORMAT}"))
        if SHOW: plt.show()

    except Exception as e:
        logging.error(f"Error plotting hyperparameter slice plots: {e}")
        raise

def plot_pca_hyperparameter_space(trials_df: pd.DataFrame,
                                  trial_max_map: Dict[int, float]) -> None:
    """
    Performs PCA on hyperparameter data and plots PC0 vs PC1, colored by mAP@50-95,
    filling unexplored areas with a mAP@50-95 value of 0.

    Args:
        trials_df (pd.DataFrame): DataFrame containing trials data, including hyperparameters.
        trial_max_map (Dict[int, float]): Mapping of trial number to max mAP@50-95.

    Returns:
        None

    Throws:
        Exception: If PCA requirements are not met or plotting fails.
    """
    try:
        import logging
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.path import Path  # Correct import for Path
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from scipy.interpolate import griddata
        from scipy.spatial import ConvexHull
        from typing import List, Dict

        # Extract hyperparameter columns
        hyperparameter_columns: List[str] = [col for col in trials_df.columns if col.startswith('params_')]

        # Select relevant columns
        trial_params_df: pd.DataFrame = trials_df[['number'] + hyperparameter_columns]

        # Add max mAP@50-95 to the DataFrame
        trial_params_df['max_mAP'] = trial_params_df['number'].map(trial_max_map)

        # Remove trials without max_mAP
        trial_params_df = trial_params_df.dropna(subset=['max_mAP'])

        # Extract hyperparameter values and standardize them
        X: np.ndarray = trial_params_df[hyperparameter_columns].values
        scaler: StandardScaler = StandardScaler()
        X_scaled: np.ndarray = scaler.fit_transform(X)

        # Perform PCA
        pca: PCA = PCA()
        X_pca: np.ndarray = pca.fit_transform(X_scaled)

        # Check if the first two PCs explain more than 95% of the variance
        explained_variance_ratio: np.ndarray = pca.explained_variance_ratio_
        cumulative_variance: float = explained_variance_ratio[:2].sum()
        if cumulative_variance < 0.95:
            logging.warning(f"The first two principal components explain {cumulative_variance:.2%} of the variance, "
                            "which is less than 95%. Consider reviewing the hyperparameter data.")
        else:
            logging.info(f"The first two principal components explain {cumulative_variance:.2%} of the variance.")

        # Prepare data for plotting
        pc0: np.ndarray = X_pca[:, 0]
        pc1: np.ndarray = X_pca[:, 1]
        max_mAP_values: np.ndarray = trial_params_df['max_mAP'].values

        # Define the grid over the PC0 and PC1 space
        grid_x, grid_y = np.meshgrid(
            np.linspace(pc0.min(), pc0.max(), 100),
            np.linspace(pc1.min(), pc1.max(), 100)
        )

        # Interpolate mAP values over the grid
        grid_z = griddata((pc0, pc1), max_mAP_values, (grid_x, grid_y), method='cubic', fill_value=0)

        # Create a convex hull around the data points to mask unexplored areas
        points: np.ndarray = np.vstack((pc0, pc1)).T
        hull: ConvexHull = ConvexHull(points)

        # Create a mask for the grid points that are outside the convex hull
        path = Path(points[hull.vertices])
        mask = ~path.contains_points(np.vstack((grid_x.ravel(), grid_y.ravel())).T).reshape(grid_x.shape)
        grid_z[mask] = 0  # Set unexplored areas to 0

        # Create the contour plot
        plt.figure(figsize=(12, 8))
        contour = plt.contourf(grid_x, grid_y, grid_z, levels=14, cmap='viridis', vmin=0.0)
        plt.colorbar(contour, label='mAP@50-95')

        # Plot the data points
        plt.scatter(pc0, pc1, c=max_mAP_values, cmap='viridis', edgecolors='k', vmin=0.0)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA of Hyperparameter Space Colored by mAP@50-95')
        plt.grid(True)
        plt.savefig(os.path.join(PATH, f'latent_space{FORMAT}'))
        if SHOW: plt.show()

    except Exception as e:
        logging.error(f"Error plotting PCA hyperparameter space: {e}")
        raise


def plot_trial_vs_map_with_optimal_line(trials_df: pd.DataFrame,
                                        trial_max_map: Dict[int, float],
                                        trial_state: Dict[int, str],
                                        best_trial_number: int,
                                        baseline_results_path: str) -> None:
    """
    Plots trial vs. mAP@50-95, with optimal solutions and baseline comparison.

    Args:
        trials_df (pd.DataFrame): DataFrame containing trials data, including hyperparameters.
        trial_max_map (Dict[int, float]): Mapping of trial number to max mAP@50-95.
        trial_state (Dict[int, str]): Mapping of trial number to state.
        best_trial_number (int): Trial number of the best trial.
        baseline_results_path (str): Path to the baseline results CSV file.

    Returns:
        None

    Throws:
        Exception: If there are issues with data processing or plotting.
    """
    try:
        import logging
        import pandas as pd
        import matplotlib.pyplot as plt
        from typing import Dict, List

        # Read the baseline results and extract the maximum mAP@50-95
        try:
            baseline_df: pd.DataFrame = pd.read_csv(baseline_results_path)
            if 'metrics/mAP50-95(B)' in baseline_df.columns:
                baseline_max_map: float = baseline_df['metrics/mAP50-95(B)'].max()
            else:
                raise ValueError("Required column 'metrics/mAP50-95(B)' not found in baseline results CSV.")
        except Exception as e:
            logging.error(f"Error reading baseline results CSV: {e}")
            raise

        # Prepare data for plotting
        trial_numbers: List[int] = trials_df['number'].tolist()
        mAP_values: List[float] = [trial_max_map[num] for num in trial_numbers]
        best_mAP: float = trial_max_map[best_trial_number]

        # Initialize lists for trial types
        completed_trials_x = []
        completed_trials_y = []
        pruned_trials_x = []
        pruned_trials_y = []

        # Categorize trials
        for num in trial_numbers:
            if num == best_trial_number:
                continue  # Skip the best trial for now
            if trial_state[num] == 'COMPLETE':
                completed_trials_x.append(num)
                completed_trials_y.append(trial_max_map[num])
            elif trial_state[num] == 'PRUNED':
                pruned_trials_x.append(num)
                pruned_trials_y.append(trial_max_map[num])

        # Create the optimal solution line
        optimal_line_x = [trial_numbers[0]]
        optimal_line_y = [mAP_values[0]]
        current_max = mAP_values[0]

        for num, value in zip(trial_numbers[1:], mAP_values[1:]):
            if value > current_max:
                current_max = value
            optimal_line_x.append(num)
            optimal_line_y.append(current_max)

        # Plotting
        plt.figure(figsize=(15, 6))
        ax = plt.subplot(111) 
        # Scatter plots for trials
        plt.scatter(pruned_trials_x, pruned_trials_y, color='grey', alpha=0.5, label='Pruned trials')
        plt.scatter(completed_trials_x, completed_trials_y, color='blue', alpha=0.5, label='Completed trials')
        plt.scatter([best_trial_number], [best_mAP], color='red', s=100, label=f'Best trial (max mAP@50-95: {best_mAP:.4f}, Trial: {best_trial_number})')

        # Optimal solutions line
        plt.plot(optimal_line_x, optimal_line_y, color='black', linestyle='-', label='Optimal solution line')

        # Baseline line
        plt.axhline(y=baseline_max_map, color='green', linestyle='--', label=f'Baseline mAP@50-95: {baseline_max_map:.4f}')

        # Labels and legend
        plt.xlabel('Trial')
        plt.ylabel('mAP@50-95')
        plt.title('Trial vs. mAP@50-95 with Optimal Solutions and Baseline Comparison')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=5)
        ax.spines[['right', 'top']].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        # Display the plot
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(PATH, f'trial_vs_map{FORMAT}'))
        plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space at the bottom for the legend
        if SHOW: plt.show()

    except Exception as e:
        logging.error(f"Error plotting trial vs. mAP@50-95: {e}")
        raise


def main():
    """
    Main function to execute the visualization script.

    Args:
        None

    Returns:
        None

    Throws:
        Exception: If any errors occur during execution.
    """
    try:
        # Replace with the actual path to your trials.csv file
        trials_csv_path: str = os.path.join(PATH, "hyperparameter_optimization_results.csv")

        # Replace with the actual path to your runs directory
        runs_dir: str = os.path.join(PATH, "runs/")

        # Read the trials CSV file
        trials_df: pd.DataFrame = read_trials_csv(trials_csv_path)

        # Create a mapping from trial number to state
        trial_state: Dict[int, str] = get_trial_state_mapping(trials_df)

        # Process trial directories
        trials_data, trial_max_map = process_trial_directories(runs_dir)

        # Identify the best trial
        best_trial_number: int = identify_best_trial(trial_max_map)

        # Compute trial statistics
        completed_percentage, pruned_percentage = compute_trial_statistics(trials_df)

        # Plot the trial data
        plot_trial_data(trials_data, trial_state, best_trial_number, trial_max_map, completed_percentage, pruned_percentage)

        # Plot the slice data
        results_baseline = os.path.join(runs_dir, "trial_1_training", "results.csv")
        baseline_results = {
            'params_lr0': 0.01,
            'params_lrf': 0.01,
            'params_momentum': 0.937,
            'params_weight_decay': 0.0005,
            'params_warmup_epochs': 3.0,
            'params_warmup_momentum': 0.8,
            'params_box': 7.5,
            'params_cls': 0.5,
            'params_dfl': 1.5,
            'results': f"{results_baseline}"
        }
        plot_hyperparameter_slice_plots(trials_df, trial_max_map, trial_state, best_trial_number, baseline_results)
        plot_pca_hyperparameter_space(trials_df, trial_max_map)
        plot_trial_vs_map_with_optimal_line(trials_df=trials_df, trial_max_map=trial_max_map, trial_state=trial_state, best_trial_number=best_trial_number, baseline_results_path=results_baseline)

    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        print("An error occurred during execution. Please check the log file for details.")

if __name__ == '__main__':
    main()
