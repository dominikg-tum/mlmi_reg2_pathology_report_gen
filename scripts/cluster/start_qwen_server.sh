#!/bin/bash
#SBATCH --job-name=qwen-server
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/qwen_server_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/qwen_server_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_paths.sh
source "${SCRIPT_DIR}/load_paths.sh"
load_cluster_paths

mkdir -p "${LOGS_DIR}"

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${CONTAINER}" \
  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${MODEL_NAME}" \
    --port 8000
