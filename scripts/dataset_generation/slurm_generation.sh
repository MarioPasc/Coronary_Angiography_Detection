#!/usr/bin/env bash
#SBATCH --job-name=CoronaryDatasetGen
#SBATCH --time=00:20:00    # Request 1 day of runtime, adjust as needed
#SBATCH --constraint=cal
#SBATCH --mem=4G            # Request 32 GB of memory, adjust based on your dataset size
#SBATCH --ntasks=1           # Run a single task
#SBATCH --cpus-per-task=1    # Request 4 CPUs for the task, adjust as needed
#SBATCH --error=dataset_gen.%J.err  # Standard error log file
#SBATCH --output=dataset_gen.%J.out # Standard output log file

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (e.g., miniconda or anaconda)
# This line might vary depending on your HPC environment
module load miniconda # Or anaconda, or specific python module

# Activate your Conda environment
# Replace 'ica_detection' with the actual name of your environment
source activate ica

# Define the path to your scripts and config file
SCRIPT_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/Coronary_Angiography_Detection/scripts/dataset_generation"
PYTHON_SCRIPT="${SCRIPT_DIR}/generate_dataset.py"
CONFIG_FILE="$./config.yaml"

# Print some information to the log
echo "Starting dataset generation script..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Python script: $PYTHON_SCRIPT"
echo "Config file: $CONFIG_FILE"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Execute the Python script
python "$PYTHON_SCRIPT" --config "$CONFIG_FILE"

echo "Script execution finished."
