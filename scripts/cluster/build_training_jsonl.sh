#!/bin/bash
#SBATCH --job-name=lora-build-jsonl
#SBATCH --chdir=/mnt/home/dogakonuk/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/home/dogakonuk/logs/lora_build_jsonl_%j.out
#SBATCH --error=/mnt/home/dogakonuk/logs/lora_build_jsonl_%j.err
#
# Build the Phase-1 LoRA SFT dataset (training/samples.jsonl) by replaying the
# Phase-1 visual pathway (thumbnail + CONCH top-k patches) over the GT chains.
#
# Owner: DOGA (fine-tuning lane). Reads the shared offline caches (configs/vision.yaml
# cache_root) READ-ONLY; per-node patch crops are written under IMAGES_OUT (yours).
#
# Prerequisites:
#   * data/labels/chains.jsonl exists
#   * offline caches exist per train slide (patch_embeddings_20x.pt + thumbnails)
#   * HF token for MahmoodLab/TITAN + CONCH (see load_paths.sh)
#
# Uses the TITAN transformers pin (4.46) — this is the DATA step, not training.
#
# Env overrides: REPO_DIR, SPLIT, LIMIT, OUTPUT, IMAGES_OUT

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/home/dogakonuk/mlmi_reg2_pathology_report_gen}"
# shellcheck source=load_paths.sh
source "${REPO_DIR}/scripts/cluster/load_paths.sh"
load_cluster_paths "${REPO_DIR}/configs/paths.yaml"

LOG_DIR="${LOG_DIR:-/mnt/home/dogakonuk/logs}"
mkdir -p "${LOG_DIR}"

SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-0}"
OUTPUT="${OUTPUT:-${REPO_DIR}/training/samples.jsonl}"
IMAGES_OUT="${IMAGES_OUT:-/mnt/home/dogakonuk/lora/train_images}"

mkdir -p "${IMAGES_OUT}"
readarray -t _ENROOT_HF_ENV < <(cluster_enroot_hf_env)

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${_ENROOT_HF_ENV[@]}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO_DIR}'
$(cluster_titan_pip_snippet)
$(cluster_hf_login_snippet)
    python -m scripts.training.build_training_jsonl \
      --output '${OUTPUT}' \
      --images-out '${IMAGES_OUT}' \
      --split '${SPLIT}' \
      --limit '${LIMIT}' \
      --visual patch_retrieve \
      --retriever graph_guided
  "
