#!/bin/bash
#SBATCH --job-name=pathology-lora-eval
#SBATCH --chdir=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/logs/lora_eval_%j.out
#SBATCH --error=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/logs/lora_eval_%j.err
#
# Evaluate the Phase-1 node answerer on the TEST split: base vs LoRA adapter.
# Owner: DOGA (fine-tuning lane). Defaults use Dominik's pinned repo / work_dir;
# override REPO_DIR / ADAPTER_DIR / LORA_ADAPTER_DIR for personal adapters.
#
# Prerequisite: a TEST-split samples file, e.g. build it with
#   SPLIT=test OUTPUT=${WORK_DIR}/lora/test_samples.jsonl \
#   IMAGES_OUT=${WORK_DIR}/lora/test_images \
#   sbatch scripts/cluster/build_training_jsonl.sh
#
# Runs the eval twice (base, then adapter) as separate processes so each gets a
# clean GPU. Compare the two OVERALL accuracies printed near the end of the log.
#
# Env overrides: REPO_DIR, TEST_JSONL, REPORT_DIR, LIMIT, MAX_NEW_TOKENS,
#                LORA_ADAPTER_DIR, ADAPTER_DIR

set -euo pipefail

PINNED_DEFAULT="/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen"
REPO_DIR="${REPO_DIR:-${MLMI_PINNED_REPO:-${PINNED_DEFAULT}}}"
# shellcheck source=load_paths.sh
source "${REPO_DIR}/scripts/cluster/load_paths.sh"
load_cluster_paths "${REPO_DIR}/configs/paths.yaml"

LOG_DIR="${LOG_DIR:-${LOGS_DIR}}"
mkdir -p "${LOG_DIR}"

TEST_JSONL="${TEST_JSONL:-${WORK_DIR}/lora/test_samples.jsonl}"
REPORT_DIR="${REPORT_DIR:-${WORK_DIR}/lora/eval}"
LIMIT="${LIMIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
BASE_MODEL="${LORA_BASE_MODEL:-${MODEL}}"
ADAPTER_DIR="${ADAPTER_DIR:-${LORA_ADAPTER_DIR:-${WORK_DIR}/lora/qwen3vl-uterus/adapter}}"

mkdir -p "${REPORT_DIR}"
readarray -t _ENROOT_HF_ENV < <(cluster_enroot_hf_env)

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${_ENROOT_HF_ENV[@]}" \
  --env USER="${USER}" --env LOGNAME="${USER}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO_DIR}'
    pip install -q 'transformers>=4.57' 'peft>=0.13' accelerate 'qwen-vl-utils>=0.0.8' pillow pyyaml
$(cluster_hf_login_snippet)
    echo '=================== BASE MODEL ==================='
    python -m scripts.training.eval_finetuned \
      --test-jsonl '${TEST_JSONL}' \
      --base-model '${BASE_MODEL}' \
      --adapter-dir '' \
      --limit '${LIMIT}' \
      --max-new-tokens '${MAX_NEW_TOKENS}' \
      --report '${REPORT_DIR}/base_report.json'
    echo '================= FINETUNED MODEL ================='
    python -m scripts.training.eval_finetuned \
      --test-jsonl '${TEST_JSONL}' \
      --base-model '${BASE_MODEL}' \
      --adapter-dir '${ADAPTER_DIR}' \
      --limit '${LIMIT}' \
      --max-new-tokens '${MAX_NEW_TOKENS}' \
      --report '${REPORT_DIR}/finetuned_report.json'
  "
