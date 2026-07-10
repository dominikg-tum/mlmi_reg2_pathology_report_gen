#!/bin/bash
#SBATCH --job-name=lora-train-qwen3vl
#SBATCH --chdir=/mnt/projects/mlmi/reg2/dogakonuk/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dogakonuk/logs/lora_train_%x_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dogakonuk/logs/lora_train_%x_%j.err
#
# LoRA fine-tune Qwen3-VL-8B on the ChainSample dataset built by build_lora_dataset.sh.
# Runs from Doga's PINNED repo; writes the adapter to Doga's own training dir.
#
# IMPORTANT: Qwen3-VL needs a RECENT transformers (the TITAN pin transformers==4.46.0
# used by the data-build job does NOT support Qwen3-VL). This job installs its own
# training stack (transformers/peft/accelerate) on top of the container torch — keep it
# in a SEPARATE container/session from the data builder.
#
# Prereqs:
#   - samples_train.jsonl + images/ from: sbatch scripts/cluster/build_lora_dataset.sh
#   - Qwen3-VL-8B weights at /mnt/projects/mlmi/reg2/models/Qwen3-VL-8B-Instruct
#   - Enough shared reg2 quota for the adapter output (~hundreds of MB).
#
# Usage:
#   sbatch scripts/cluster/train_lora.sh
#   EPOCHS=1 LIMIT_JSONL=... sbatch scripts/cluster/train_lora.sh   # (see vars below)

set -euo pipefail

USER_ROOT="${MLMI_USER_ROOT:-/mnt/projects/mlmi/reg2/dogakonuk}"
PINNED_REPO="${MLMI_PINNED_REPO:-${USER_ROOT}/repos/mlmi_reg2_pathology_report_gen}"

# shellcheck source=load_paths.sh
source "${PINNED_REPO}/scripts/cluster/load_paths.sh"
load_cluster_paths
REPO="${PINNED_REPO}"
LOGS_DIR="${USER_ROOT}/logs"
mkdir -p "${LOGS_DIR}"

TRAIN_JSONL="${TRAIN_JSONL:-${USER_ROOT}/training/samples_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${USER_ROOT}/training/lora/qwen3vl_8b_v1}"
BASE_MODEL="${BASE_MODEL:-${MODEL:-/mnt/projects/mlmi/reg2/models/Qwen3-VL-8B-Instruct}}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
RANK="${RANK:-16}"

mkdir -p "${OUTPUT_DIR}"

CONTAINER="${PERSONAL_CONTAINER:-${CONTAINER:-/mnt/projects/mlmi/reg2/containers/qwen25_dev_updated.sqsh}}"
readarray -t _ENROOT_HF_ENV < <(cluster_enroot_hf_env)

CMD=(
  python -m training.lora
  --train-jsonl "${TRAIN_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --base-model "${BASE_MODEL}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM}"
  --lr "${LR}"
  --rank "${RANK}"
)
printf -v INNER_CMD '%q ' "${CMD[@]}"
INNER_CMD="${INNER_CMD% }"

echo "REPO=${REPO}"
echo "TRAIN_JSONL=${TRAIN_JSONL}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE} GRAD_ACCUM=${GRAD_ACCUM} LR=${LR} RANK=${RANK}"
echo "CONTAINER=${CONTAINER}"

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${_ENROOT_HF_ENV[@]}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    # Qwen3-VL training stack — recent transformers (NOT the TITAN 4.46.0 pin).
    pip install -q -U 'transformers>=4.57' peft accelerate 'safetensors>=0.4' 2>/dev/null || \
      pip install -q -U transformers peft accelerate safetensors
    pip install -q pillow pyyaml 2>/dev/null || true
$(cluster_hf_login_snippet)
    ${INNER_CMD}
  "
