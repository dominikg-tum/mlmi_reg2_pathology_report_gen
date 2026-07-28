#!/bin/bash
#SBATCH --job-name=qwen-server
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/qwen_server_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/qwen_server_%j.err

set -euo pipefail

# shellcheck source=load_paths.sh
source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths

mkdir -p "${LOGS_DIR}"

readarray -t _ENROOT_HF_ENV < <(cluster_enroot_hf_env)

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${_ENROOT_HF_ENV[@]}" \
  "${CONTAINER}" \
  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${MODEL_NAME}" \
    --port 8000
