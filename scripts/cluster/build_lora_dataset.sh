#!/bin/bash
#SBATCH --job-name=lora-dataset-build
#SBATCH --chdir=/mnt/projects/mlmi/reg2/dogakonuk/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dogakonuk/logs/lora_dataset_%x_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dogakonuk/logs/lora_dataset_%x_%j.err
#
# Build the LoRA training-data JSONL from WP3 chains (train split) using the SAME
# visual pathway as inference: patch_retrieve + graph_guided (TITAN text encoder over
# the precomputed CONCH patch pools). Runs from Doga's PINNED repo and writes to Doga's
# own training dir — NOT the shared team checkout, NOT dominik's folder.
#
# Prereqs (already satisfied):
#   - chains.jsonl in the shared team repo (gitignored, not in the pinned clone).
#   - Shared read-only cache with patch_embeddings_{5x,10x,20x}.pt + thumbnails
#     (configs/vision.yaml -> cache_root = /mnt/projects/mlmi/reg2/dominik/cache).
#   - HF token for MahmoodLab/TITAN in ~/.hf_env  (echo 'export HF_TOKEN=hf_...' > ~/.hf_env; chmod 600 ~/.hf_env)
#
# Usage:
#   sbatch scripts/cluster/build_lora_dataset.sh                 # full train split, patch_retrieve + TITAN
#   LIMIT=20 sbatch scripts/cluster/build_lora_dataset.sh        # first 20 train slides (quick real run)
#   SMOKE=1 sbatch scripts/cluster/build_lora_dataset.sh         # 3-slide thumbnail smoke (no TITAN)
#   SEARCH_ALL_PATCHES=1 sbatch scripts/cluster/build_lora_dataset.sh   # full patch pool (no k-means prefilter)

set -euo pipefail

USER_ROOT="${MLMI_USER_ROOT:-/mnt/projects/mlmi/reg2/dogakonuk}"
PINNED_REPO="${MLMI_PINNED_REPO:-${USER_ROOT}/repos/mlmi_reg2_pathology_report_gen}"

# shellcheck source=load_paths.sh
source "${PINNED_REPO}/scripts/cluster/load_paths.sh"
load_cluster_paths          # provides pip/HF helper functions + CONTAINER default
REPO="${PINNED_REPO}"       # override team REPO from paths.yaml with the pinned clone
LOGS_DIR="${USER_ROOT}/logs"
mkdir -p "${LOGS_DIR}"

# chains.jsonl is gitignored -> only exists in the TEAM repo, not the pinned clone.
CHAINS="${CHAINS:-/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/data/labels/chains.jsonl}"

SPLIT="${SPLIT:-train}"
VISUAL="${VISUAL:-patch_retrieve}"
RETRIEVER="${RETRIEVER:-graph_guided}"
PRIMARY_INDEX="${PRIMARY_INDEX:-1}"
SEARCH_ALL_PATCHES="${SEARCH_ALL_PATCHES:-0}"

OUT_DIR="${USER_ROOT}/training"
mkdir -p "${OUT_DIR}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  VISUAL="thumbnail"
  RETRIEVER="none"
  LIMIT="${LIMIT:-3}"
  OUTPUT="${OUTPUT:-${OUT_DIR}/samples_smoke.jsonl}"
else
  LIMIT="${LIMIT:-0}"
  OUTPUT="${OUTPUT:-${OUT_DIR}/samples_${SPLIT}.jsonl}"
fi
IMAGE_ROOT="${IMAGE_ROOT:-${OUT_DIR}/images}"

CONTAINER="${PERSONAL_CONTAINER:-${CONTAINER:-/mnt/projects/mlmi/reg2/containers/qwen25_dev_updated.sqsh}}"
readarray -t _ENROOT_HF_ENV < <(cluster_enroot_hf_env)

CMD=(
  python -m training.dataset
  --chains "${CHAINS}"
  --output "${OUTPUT}"
  --image-root "${IMAGE_ROOT}"
  --split "${SPLIT}"
  --visual "${VISUAL}"
  --retriever "${RETRIEVER}"
  --primary-index "${PRIMARY_INDEX}"
)
if [[ "${LIMIT}" != "0" ]]; then
  CMD+=(--limit "${LIMIT}")
fi
if [[ "${SEARCH_ALL_PATCHES}" == "1" ]]; then
  CMD+=(--search-all-patches)
fi

printf -v INNER_CMD '%q ' "${CMD[@]}"
INNER_CMD="${INNER_CMD% }"

echo "REPO=${REPO}"
echo "CHAINS=${CHAINS}"
echo "OUTPUT=${OUTPUT}"
echo "IMAGE_ROOT=${IMAGE_ROOT}"
echo "VISUAL=${VISUAL} RETRIEVER=${RETRIEVER} SPLIT=${SPLIT} LIMIT=${LIMIT}"
echo "CONTAINER=${CONTAINER}"

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${_ENROOT_HF_ENV[@]}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
$(cluster_titan_pip_snippet)
$(cluster_hf_login_snippet)
    ${INNER_CMD}
  "
