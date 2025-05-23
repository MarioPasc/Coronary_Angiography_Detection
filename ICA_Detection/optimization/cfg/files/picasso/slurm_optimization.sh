#!/usr/bin/env bash
#
#SBATCH --job-name=BHO_GP_ARCADE_preproc
#SBATCH --time=3-00:00:00          # 3 days
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4               # request 4 GPUs
#SBATCH --constraint=dgx           # DGX node
#SBATCH --error=log_picasso_bho.%J.err
#SBATCH --output=log_picasso_bho.%J.out

########################################
# 1) User settings
########################################
CONFIG_YAML=""       # <<<<  EDIT  <<<
ENV_NAME="ica"                            

########################################
# 2) Software environment
########################################
module load miniconda
source activate "${ENV_NAME}"

########################################
# 3) Build the --gpu-ids argument
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
# 4) Launch the optimisation
########################################
# run_optimization --config "${CONFIG_YAML}" --gpu-ids "${GPU_IDS}"
run_optimization --config "${CONFIG_YAML}" --gpu-ids -1
