#!/bin/bash
#SBATCH --job-name=wp3-text-server
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/wp3_server_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/wp3_server_%j.err
#
# Text-only vLLM for WP3 CoT extraction (lighter than Qwen3-VL on 1 GPU).
# Uses medgemma-1.5-4b-it — already on disk at configs/paths.yaml → models.medgemma_4b.
#
# After start, set in configs/paths.yaml (or env for one-off runs):
#   qwen.api_base_url: http://<node>:8000/v1
#   qwen.model_name: google/medgemma-1.5-4b-it   # or path basename served name
#
# Usage:
#   sbatch scripts/cluster/start_wp3_server.sh
#   # or override model:
#   WP3_MODEL=/mnt/projects/mlmi/reg2/models/medgemma-1.5-4b-it sbatch scripts/cluster/start_wp3_server.sh

set -euo pipefail

source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths

mkdir -p "${LOGS_DIR}"

WP3_MODEL="${WP3_MODEL:-/mnt/projects/mlmi/reg2/models/medgemma-1.5-4b-it}"
WP3_MODEL_NAME="${WP3_MODEL_NAME:-medgemma-1.5-4b-it}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

readarray -t _ENROOT_HF_ENV < <(cluster_enroot_hf_env)

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${_ENROOT_HF_ENV[@]}" \
  "${CONTAINER}" \
  python -m vllm.entrypoints.openai.api_server \
    --model "${WP3_MODEL}" \
    --served-model-name "${WP3_MODEL_NAME}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --port 8000
