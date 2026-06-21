#!/bin/bash
#SBATCH --job-name=pathology-patch-retrieve-baseline
#
# Prerequisite: long-running Qwen vLLM server (NOT started by this script).
#   sbatch scripts/cluster/start_qwen_server.sh
#   squeue -u $USER   # note the node name, e.g. heidelberg
#   curl -s http://<node>:8000/v1/models | head
# Then submit with matching API base (default assumes heidelberg):
#   QWEN_API_BASE=http://heidelberg:8000/v1 SEARCH_ALL_PATCHES=1 FORCE=1 sbatch scripts/cluster/run_patch_retrieve_baseline.sh
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/patch_retrieve_baseline_%x_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/patch_retrieve_baseline_%x_%j.err

set -euo pipefail

BACKEND="${BACKEND:-qwen}"
LIMIT="${LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
SEARCH_ALL_PATCHES="${SEARCH_ALL_PATCHES:-0}"
FORCE="${FORCE:-0}"
QWEN_API_BASE="${QWEN_API_BASE:-http://heidelberg:8000/v1}"
DEPENDENCY="${DEPENDENCY:-}"

# shellcheck source=load_paths.sh
source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths

mkdir -p "${LOGS_DIR}"

if [[ "${SMOKE}" == "1" ]]; then
  RUNS_SUBDIR="baseline_patch_retrieve_smoke"
  PRED_NAME="predictions_test_patch_retrieve_smoke.jsonl"
  LIMIT="${LIMIT:-3}"
elif [[ "${SEARCH_ALL_PATCHES}" == "1" ]]; then
  RUNS_SUBDIR="baseline_patch_retrieve_fullpool"
  PRED_NAME="predictions_test_baseline_patch_retrieve_fullpool.jsonl"
else
  RUNS_SUBDIR="baseline_patch_retrieve"
  PRED_NAME="predictions_test_baseline_patch_retrieve.jsonl"
fi

CONTAINER="${PERSONAL_CONTAINER:-/mnt/projects/mlmi/reg2/containers/dominik_20260529_base.sqsh}"
readarray -t _ENROOT_HF_ENV < <(cluster_enroot_hf_env)

CMD=(
  python -m scripts.inference.run_test_baseline_batch
  --backend "${BACKEND}"
  --split test
  --visual patch_retrieve
  --retriever graph_guided
  --navigator graph_guided
  --runs-subdir "${RUNS_SUBDIR}"
  --predictions "${WORK_DIR}/runs/${PRED_NAME}"
  --require-embeddings
)
if [[ "${SEARCH_ALL_PATCHES}" == "1" ]]; then
  CMD+=(--search-all-patches)
fi
if [[ "${FORCE}" == "1" ]]; then
  CMD+=(--force)
else
  CMD+=(--skip-existing)
fi
if [[ "${LIMIT}" != "0" ]]; then
  CMD+=(--limit "${LIMIT}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry-run)
fi

printf -v INNER_CMD '%q ' "${CMD[@]}"
INNER_CMD="${INNER_CMD% }"

SBATCH_EXTRA=()
if [[ -n "${DEPENDENCY}" ]]; then
  SBATCH_EXTRA+=(--dependency="afterok:${DEPENDENCY}")
fi

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${_ENROOT_HF_ENV[@]}" \
  --env "QWEN_API_BASE=${QWEN_API_BASE}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
$(cluster_titan_pip_snippet)
    pip install -q openai pandas openpyxl pyyaml 2>/dev/null || true
    python -m scripts.data.build_wsi_id_map
    ${INNER_CMD}
    if [[ '${DRY_RUN}' != '1' ]]; then
      python -m eval.run_eval \
        --pred '${WORK_DIR}/runs/${PRED_NAME}' \
        --gt '${REPO}/data/labels/chains.jsonl' \
        --split test
    fi
  "
