#!/bin/bash
#SBATCH --job-name=lora-train
#SBATCH --chdir=/mnt/home/dogakonuk/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/dogakonuk/logs/lora_train_%j.out
#SBATCH --error=/mnt/home/dogakonuk/logs/lora_train_%j.err
#
# LoRA fine-tune the Phase-1 node answerer (Qwen3-VL-8B) on training/samples.jsonl.
# Owner: DOGA (fine-tuning lane).
#
# Prerequisite: training/samples.jsonl (scripts/cluster/build_training_jsonl.sh).
#
# IMPORTANT: installs transformers>=4.57 for Qwen3-VL — do NOT reuse the TITAN pin
# env; run this as a separate job from the data build.
#
# Env overrides: REPO_DIR, EPOCHS, LR, LORA_R, BATCH_SIZE, GRAD_ACCUM,
#                LOAD_IN_4BIT=1, TRAIN_JSONL, OUTPUT_DIR

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/home/dogakonuk/mlmi_reg2_pathology_report_gen}"
# shellcheck source=load_paths.sh
source "${REPO_DIR}/scripts/cluster/load_paths.sh"
load_cluster_paths "${REPO_DIR}/configs/paths.yaml"

LOG_DIR="${LOG_DIR:-/mnt/home/dogakonuk/logs}"
mkdir -p "${LOG_DIR}"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
LORA_R="${LORA_R:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
TRAIN_JSONL="${TRAIN_JSONL:-${REPO_DIR}/training/samples.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${LORA_ADAPTER_DIR:-/mnt/home/dogakonuk/lora/qwen3vl-uterus/adapter}}"

FOURBIT_FLAG=""
if [[ "${LOAD_IN_4BIT:-0}" == "1" ]]; then
  FOURBIT_FLAG="--load-in-4bit"
fi

mkdir -p "${OUTPUT_DIR}"
readarray -t _ENROOT_HF_ENV < <(cluster_enroot_hf_env)

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${_ENROOT_HF_ENV[@]}" \
  --env USER="${USER:-dogakonuk}" --env LOGNAME="${USER:-dogakonuk}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO_DIR}'
    pip install -q 'transformers>=4.57' 'peft>=0.13' 'trl>=0.12' accelerate datasets bitsandbytes 'qwen-vl-utils>=0.0.8' pillow pyyaml
$(cluster_hf_login_snippet)
    python -m scripts.training.run_lora \
      --train-jsonl '${TRAIN_JSONL}' \
      --output-dir '${OUTPUT_DIR}' \
      --base-model '${LORA_BASE_MODEL:-${MODEL}}' \
      --epochs '${EPOCHS}' \
      --lr '${LR}' \
      --lora-r '${LORA_R}' \
      --batch-size '${BATCH_SIZE}' \
      --grad-accum '${GRAD_ACCUM}' \
      ${FOURBIT_FLAG}
  "
