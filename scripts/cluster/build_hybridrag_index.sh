#!/bin/bash
#SBATCH --job-name=pathology-build-hybridrag
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/build_hybridrag_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/build_hybridrag_%j.err
#
# Build HybridRAG Chroma + BM25 index from train-split labels xlsx.
#
# Usage:
#   sbatch scripts/cluster/build_hybridrag_index.sh
#   FORCE_REBUILD=1 sbatch scripts/cluster/build_hybridrag_index.sh

set -euo pipefail

SPLIT="${SPLIT:-train}"
FORCE_REBUILD="${FORCE_REBUILD:-}"

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
$(cluster_hybridrag_pip_snippet)
    ARGS=(python -m scripts.memory.build_hybridrag_index --split '${SPLIT}')
    [[ -n '${FORCE_REBUILD}' ]] && ARGS+=(--force-rebuild)
    \"\${ARGS[@]}\"
  "
