#!/bin/bash
#SBATCH --job-name=pathology-build-hipporag
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/build_hipporag_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/build_hipporag_%j.err
#
# Build HippoRAG2 embedding index from train-split chains.jsonl.
#
# Usage:
#   sbatch scripts/cluster/build_hipporag_index.sh
#   SPLIT=train sbatch scripts/cluster/build_hipporag_index.sh

set -euo pipefail

SPLIT="${SPLIT:-train}"

# shellcheck source=load_paths.sh
source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths

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
