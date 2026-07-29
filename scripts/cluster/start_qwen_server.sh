#!/bin/bash
#SBATCH --job-name=qwen-server
#SBATCH --chdir=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=NONE
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/logs/qwen_server_%j.out
#SBATCH --error=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/logs/qwen_server_%j.err
#
# Starts OpenAI-compatible Qwen3-VL-8B via vLLM. Baseline jobs must reach this
# node: either co-locate, or set qwen.api_base_url to http://<node>:8000/v1.
#
# Usage:
#   sbatch scripts/cluster/start_qwen_server.sh
#   # URL file is written only after /v1/models answers (not at job start).

set -euo pipefail

export PATH="/usr/local/bin:/usr/bin:/bin${PATH:+:${PATH}}"

PINNED_REPO="${MLMI_PINNED_REPO:-/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen}"
export MLMI_PINNED_REPO="${PINNED_REPO}"

# shellcheck source=load_paths.sh
source "${PINNED_REPO}/scripts/cluster/load_paths.sh"
load_cluster_paths "${PINNED_REPO}/configs/paths.yaml"
REPO="${PINNED_REPO}"

mkdir -p "${LOGS_DIR}"

HOST="$(hostname -s 2>/dev/null || hostname)"
FQDN="${HOST}.garching.camp.cluster"
URL_FILE="${LOGS_DIR}/qwen_server_url.txt"
# Drop stale URL so baseline jobs cannot hit a dead previous host.
rm -f "${URL_FILE}"

echo "Qwen vLLM starting on ${FQDN}:8000"
echo "REPO=${REPO}"
echo "MODEL=${MODEL}"
echo "Will write ${URL_FILE} only after /v1/models is healthy"

readarray -t _ENROOT_HF_ENV < <(cluster_enroot_hf_env)

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${_ENROOT_HF_ENV[@]}" \
  "${CONTAINER}" \
  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${MODEL_NAME}" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 &
ENROOT_PID=$!

READY=0
for _ in $(seq 1 180); do
  if ! kill -0 "${ENROOT_PID}" 2>/dev/null; then
    wait "${ENROOT_PID}" || true
    echo "vLLM/enroot exited before becoming ready" >&2
    exit 1
  fi
  if curl -sf --connect-timeout 2 "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1; then
    echo "http://${FQDN}:8000/v1" > "${URL_FILE}"
    echo "Wrote ready URL to ${URL_FILE}"
    READY=1
    break
  fi
  sleep 5
done

if [[ "${READY}" -ne 1 ]]; then
  echo "Timed out waiting for vLLM /v1/models; killing enroot pid=${ENROOT_PID}" >&2
  kill "${ENROOT_PID}" 2>/dev/null || true
  wait "${ENROOT_PID}" || true
  exit 1
fi

wait "${ENROOT_PID}"
