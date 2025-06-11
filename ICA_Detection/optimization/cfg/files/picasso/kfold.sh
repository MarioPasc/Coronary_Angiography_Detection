#!/usr/bin/env bash
#
#SBATCH --job-name=kfold_cv
#SBATCH --time=3-00:00:00          # 3 days
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4               # request 4 GPUs
#SBATCH --constraint=dgx           # DGX node
#SBATCH --error=log_kfold_cv.%J.err
#SBATCH --output=log_kfold_cv.%J.out

########################################
# 1) User settings
########################################
# Path to your kfold_config.yaml file
CONFIG_YAML=""       # <<<<  EDIT  <<<
ENV_NAME="ica"                            

# K-fold splitting parameters
META_JSON="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/angio/tasks/stenosis_detection/json/processed.json"         # <<<<  EDIT  <<<< Path to original metadata JSON
KFOLD_OUTDIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/angio/tasks/stenosis_detection/kfold"  # <<<<  EDIT  <<<< Output directory for splits
K=3                  # Number of folds
SEED=42              # Random seed

# K-fold validation parameters
BASE_DIR="/tmp"
OUT_CSV="/mnt/home/users/tic_163_uma/mpascual/execs/ICA/tests/kfold/results.csv"
WORKSPACE_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/ICA/tests/kfold/kfold_results"
REMOVE_WEIGHTS=false
VERBOSE=true

########################################
# 2) Software environment
########################################
module load miniconda
source activate "${ENV_NAME}"

########################################
# 3) Create K-fold dataset splits
########################################
echo "[INFO] Running K-fold dataset splitting..."

if [[ -z "${META_JSON}" ]]; then
  echo "[ERROR] META_JSON is required for K-fold splitting. Abort." >&2
  exit 1
fi

python -m ICA_Detection.splits.k_fold.cadica.pipeline \
  --meta "${META_JSON}" \
  --out "${KFOLD_OUTDIR}" \
  --k "${K}" \
  --seed "${SEED}"

if [[ $? -ne 0 ]]; then
  echo "[ERROR] K-fold splitting failed. Abort." >&2
  exit 1
fi

echo "[INFO] K-fold splitting completed."
# The KFOLD_DIR is now the output directory from the splitting step
KFOLD_DIR="${KFOLD_OUTDIR}"

########################################
# 4) Build the --gpu-ids argument
########################################
# Slurm exports which GPUs you may use in *one* of these vars.
# We try them in order of preference.
GPUS_VAR="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-}}}"

if [[ -z "${GPUS_VAR}" ]]; then
  echo "[ERROR] Cannot detect allocated GPUs. Abort." >&2
  exit 1
fi

# Convert possible space-separated list to comma-separated (PyTorch style)
GPU_IDS=$(echo "${GPUS_VAR}" | tr -d ' ' | tr ',' '\n' | paste -sd ',' -)

echo "[INFO] Slurm granted GPUs: ${GPU_IDS}"

########################################
# 5) Launch k-fold cross-validation
########################################
echo "[INFO] Running K-fold validation..."

if [[ -n "${CONFIG_YAML}" ]]; then
    # If config file is provided, use it
    echo "[INFO] Running with config file: ${CONFIG_YAML}"
    python -m ICA_Detection.optimization.engine.kfold_validation.engine --config "${CONFIG_YAML}"
else
    # Otherwise use CLI parameters
    echo "[INFO] Running with CLI parameters"
    
    # Build arguments
    ARGS="--base-dir ${BASE_DIR} --kfold-dir ${KFOLD_DIR} --gpu-ids ${GPU_IDS} --out ${OUT_CSV} --workspace-dir ${WORKSPACE_DIR}"
    
    # Add optional flags
    if [[ "${REMOVE_WEIGHTS}" = true ]]; then
        ARGS="${ARGS} --remove-weights"
    fi
    
    if [[ "${VERBOSE}" = true ]]; then
        ARGS="${ARGS} --verbose"
    fi
    
    # Execute
    python -m ICA_Detection.optimization.engine.kfold_validation.engine ${ARGS}
fi

echo "[INFO] K-fold cross-validation completed"