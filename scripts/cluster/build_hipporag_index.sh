#!/bin/bash
#SBATCH --job-name=pathology-build-hipporag
#SBATCH --chdir=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=NONE
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/logs/build_hipporag_%j.out
#SBATCH --error=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/logs/build_hipporag_%j.err
#
# Build HippoRAG2 embedding index from train-split chains.jsonl
# (splits must be restamped from cases.csv).
#
# Usage:
#   sbatch --export=NONE,SPLIT=train scripts/cluster/build_hipporag_index.sh

set -euo pipefail

export PATH="/usr/local/bin:/usr/bin:/bin${PATH:+:${PATH}}"

PINNED_REPO="${MLMI_PINNED_REPO:-/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen}"
export MLMI_PINNED_REPO="${PINNED_REPO}"

SPLIT="${SPLIT:-train}"

# shellcheck source=load_paths.sh
source "${PINNED_REPO}/scripts/cluster/load_paths.sh"
load_cluster_paths "${PINNED_REPO}/configs/paths.yaml"
REPO="${PINNED_REPO}"

mkdir -p "${LOGS_DIR}"

CONTAINER="${PERSONAL_CONTAINER:-${CONTAINER}}"

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    pip install -q sentence-transformers pyyaml 2>/dev/null || true
    python -m scripts.memory.build_hipporag_index --split '${SPLIT}'
  "
