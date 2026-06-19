#!/bin/bash
#SBATCH --job-name=pathology-test-thumbnail-baseline
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:0
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/test_thumbnail_baseline_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/test_thumbnail_baseline_%j.err

set -euo pipefail

BACKEND="${BACKEND:-qwen}"
LIMIT="${LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
QWEN_API_BASE="${QWEN_API_BASE:-http://heidelberg:8000/v1}"

# shellcheck source=load_paths.sh
source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths

mkdir -p "${LOGS_DIR}"

CONTAINER="${PERSONAL_CONTAINER:-/mnt/projects/mlmi/reg2/containers/dominik_20260529_base.sqsh}"
readarray -t _ENROOT_HF_ENV < <(cluster_enroot_hf_env)

CMD=(python -m scripts.inference.run_test_baseline_batch --backend "${BACKEND}" --split test --skip-existing)
if [[ "${LIMIT}" != "0" ]]; then
  CMD+=(--limit "${LIMIT}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry-run)
fi

printf -v INNER_CMD '%q ' "${CMD[@]}"
INNER_CMD="${INNER_CMD% }"

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${_ENROOT_HF_ENV[@]}" \
  --env "QWEN_API_BASE=${QWEN_API_BASE}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    pip install -q openai pandas openpyxl pyyaml 2>/dev/null || true
    python -m scripts.data.build_wsi_id_map
    ${INNER_CMD}
  "
