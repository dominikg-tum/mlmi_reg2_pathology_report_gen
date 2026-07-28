#!/bin/bash
#SBATCH --job-name=pathology-baseline-batch
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:0
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/baseline_batch_%x_%A_%a.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/baseline_batch_%x_%A_%a.err
#
# Thumbnail baseline batch over test split (one slide per array task).
#
# Prerequisite: vLLM Qwen server running (sbatch scripts/cluster/start_qwen_server.sh)
# and configs/paths.yaml qwen.api_base_url reachable from compute nodes.
#
# Usage:
#   # 1) Write slide list and count tasks (N = lines - 1 for --array=0-N)
#   python -m scripts.inference.run_baseline_batch --baseline a --split test \
#     --write-slide-list /mnt/projects/mlmi/reg2/dominik/logs/baseline_test_slides.txt --dry-run
#
#   # 2) Submit array (example: 70 test slides -> --array=0-69)
#   BASELINE=a sbatch --array=0-69 scripts/cluster/run_baseline_batch.sh
#
#   BASELINE=b1 sbatch --array=0-69 scripts/cluster/run_baseline_batch.sh
#   BASELINE=b2 sbatch --array=0-69 scripts/cluster/run_baseline_batch.sh
#
# Env:
#   BASELINE=a|b1|b2   (default: a)
#   SPLIT=test         (default: test)
#   SLIDE_ID=CASE.svs  optional single-slide override (non-array)

set -euo pipefail

BASELINE="${BASELINE:-a}"
SPLIT="${SPLIT:-test}"

# shellcheck source=load_paths.sh
source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths

mkdir -p "${LOGS_DIR}"

CONTAINER="${PERSONAL_CONTAINER:-${CONTAINER}}"

HYBRID_PIP=""
if [[ "${BASELINE}" == "b2" ]]; then
  HYBRID_PIP="$(cluster_hybridrag_pip_snippet)"
fi

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    pip install -q openai pyyaml pandas openpyxl 2>/dev/null || true
${HYBRID_PIP}
    ARGS=(
      python -m scripts.inference.run_baseline_batch
      --baseline '${BASELINE}'
      --split '${SPLIT}'
      --skip-existing
    )
    if [[ -n '${SLIDE_ID:-}' ]]; then
      ARGS+=(--slide-id '${SLIDE_ID}')
    elif [[ -n '${SLURM_ARRAY_TASK_ID:-}' ]]; then
      ARGS+=(--slide-index '${SLURM_ARRAY_TASK_ID}')
    else
      echo 'Set SLURM_ARRAY_TASK_ID (array job) or SLIDE_ID=CASE.svs' >&2
      exit 1
    fi
    \"\${ARGS[@]}\"
  "
