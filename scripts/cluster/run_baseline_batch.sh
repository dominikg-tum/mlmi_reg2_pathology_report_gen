#!/bin/bash
#SBATCH --job-name=pathology-baseline-batch
#SBATCH --chdir=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen
# --export=ALL, not NONE: NONE makes slurmd fetch the user env via `su - user`,
# which fails under head-node process pressure and parks the job as
# "user env retrieval failed requeued held". The script re-exports what it needs.
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
# Request a GPU even though the baseline client is CPU-side (Qwen is a
# separate job). Campus opportunistic jobs without --gres get stuck as
# "user env retrieval failed requeued held".
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/logs/baseline_batch_%x_%j.out
#SBATCH --error=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/logs/baseline_batch_%x_%j.err
#
# Thumbnail / patch baseline batch over test split (one CASE per array task).
# Each case runs SS-LLM Pick: Phase 1 per WSI, then one selected case chain.
#
# Prerequisite: vLLM Qwen server running (sbatch scripts/cluster/start_qwen_server.sh)
# and logs/qwen_server_url.txt pointing at that node (written by the server job).
#
# Usage:
#   # Smoke one case by key:
#   sbatch --export=ALL,BASELINE=a,SPLIT=test,BACKEND=qwen,SLIDE_ID='CASE.svs,...',MLMI_PINNED_REPO=... \
#     scripts/cluster/run_baseline_batch.sh
#
#   # Full test array (count test cases in stamped chains.jsonl, then --array=0-(N-1)):
#   # After cases.csv restamp expect 70 test cases -> --array=0-69.
#   # Prefer: bash scripts/cluster/submit_baseline_batch.sh --baseline a
#   # which derives END from the live case count.
#   BASELINE=a sbatch --job-name=path-baseline-a-test --array=0-69 \
#     --export=ALL,BASELINE=a,SPLIT=test,MLMI_PINNED_REPO=... \
#     scripts/cluster/run_baseline_batch.sh
#
# Env:
#   BASELINE=a|b1|b2|b2_cap|p0|p1|p2|p3|naive   (default: a)
#   SPLIT=test                           (default: test)
#   SLIDE_ID=<case_key>                  optional single-case override (GT slide_id string)
#   BACKEND=qwen|dummy|finetuned         (default: qwen)

set -euo pipefail

export PATH="/usr/local/bin:/usr/bin:/bin${PATH:+:${PATH}}"

PINNED_REPO="${MLMI_PINNED_REPO:-/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen}"
export MLMI_PINNED_REPO="${PINNED_REPO}"

BASELINE="${BASELINE:-a}"
SPLIT="${SPLIT:-test}"
BACKEND="${BACKEND:-qwen}"
SLIDE_ID="${SLIDE_ID:-}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
# Optional numeric index when SLIDE_ID cannot be passed via --export (commas break SLURM export).
SLIDE_INDEX="${SLIDE_INDEX:-}"

# shellcheck source=load_paths.sh
source "${PINNED_REPO}/scripts/cluster/load_paths.sh"
load_cluster_paths "${PINNED_REPO}/configs/paths.yaml"
REPO="${PINNED_REPO}"

mkdir -p "${LOGS_DIR}"

# Prefer the live server URL file over paths.yaml localhost.
if [[ -f "${LOGS_DIR}/qwen_server_url.txt" ]]; then
  QWEN_API_BASE_URL="$(tr -d '[:space:]' < "${LOGS_DIR}/qwen_server_url.txt")"
  export QWEN_API_BASE_URL
fi
if [[ -n "${QWEN_API_BASE_URL:-}" ]]; then
  echo "Using QWEN_API_BASE_URL=${QWEN_API_BASE_URL}"
else
  echo "WARNING: no QWEN_API_BASE_URL / qwen_server_url.txt; client will use paths.yaml localhost" >&2
fi

CONTAINER="${PERSONAL_CONTAINER:-${CONTAINER}}"

HYBRID_PIP=""
if [[ "${BASELINE}" == "b2" || "${BASELINE}" == "b2_cap" ]]; then
  HYBRID_PIP="$(cluster_hybridrag_pip_snippet)"
fi

TITAN_PIP=""
if [[ "${BASELINE}" == p0 || "${BASELINE}" == p1 || "${BASELINE}" == p2 || "${BASELINE}" == p3 ]]; then
  TITAN_PIP="$(cluster_titan_pip_snippet)"
fi

# Build argv outside enroot so set -u / array vs SLIDE_ID is handled here.
ARGS_FILE="$(mktemp "${TMPDIR:-/tmp}/baseline_args.XXXXXX")"
{
  echo "--baseline"
  echo "${BASELINE}"
  echo "--split"
  echo "${SPLIT}"
  echo "--backend"
  echo "${BACKEND}"
  echo "--skip-existing"
  if [[ -n "${SLIDE_ID}" ]]; then
    echo "--slide-id"
    echo "${SLIDE_ID}"
  elif [[ -n "${ARRAY_TASK_ID}" ]]; then
    echo "--slide-index"
    echo "${ARRAY_TASK_ID}"
  elif [[ -n "${SLIDE_INDEX}" ]]; then
    echo "--slide-index"
    echo "${SLIDE_INDEX}"
  else
    echo "Set SLURM_ARRAY_TASK_ID, SLIDE_INDEX, or SLIDE_ID=<case_key>" >&2
    exit 1
  fi
} > "${ARGS_FILE}"

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  --env "QWEN_API_BASE_URL=${QWEN_API_BASE_URL:-}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    pip install -q openai pyyaml pandas openpyxl
${HYBRID_PIP}
${TITAN_PIP}
    mapfile -t ARGS < '${ARGS_FILE}'
    python -m scripts.inference.run_baseline_batch \"\${ARGS[@]}\"
  "
rm -f "${ARGS_FILE}"
