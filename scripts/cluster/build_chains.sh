#!/bin/bash
#SBATCH --job-name=pathology-build-chains
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:0
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/build_chains_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/build_chains_%j.err
#
# Prerequisite: vLLM Qwen server running (sbatch scripts/cluster/start_qwen_server.sh)
# and configs/paths.yaml qwen.api_base_url reachable from this node.

set -euo pipefail

# shellcheck source=load_paths.sh
source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths

mkdir -p "${LOGS_DIR}"

LIMIT="${LIMIT:-0}"
SLIDE="${SLIDE:-}"
RESUME="${RESUME:-}"
RETRY_FAILED="${RETRY_FAILED:-}"
# Remote vLLM (when not colocated on same node), e.g.:
#   VLLM_HOST=heidelberg.garching.camp.cluster RESUME=1 sbatch scripts/cluster/build_chains.sh
VLLM_HOST="${VLLM_HOST:-localhost}"
export QWEN_API_BASE_URL="${QWEN_API_BASE_URL:-http://${VLLM_HOST}:8000/v1}"

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  --env QWEN_API_BASE_URL="${QWEN_API_BASE_URL}" \
  --env QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    pip install -q openai pandas pyyaml tqdm openpyxl 2>/dev/null || true
    ARGS=(python -m scripts.extraction.build_chains_from_graph)
    [[ -n '${SLIDE}' ]] && ARGS+=(--slide '${SLIDE}')
    [[ '${LIMIT}' != '0' ]] && ARGS+=(--limit '${LIMIT}')
    [[ -n '${RESUME}' ]] && ARGS+=(--resume)
    [[ -n '${RETRY_FAILED}' ]] && ARGS+=(--retry-failed)
    \"\${ARGS[@]}\"
  "
