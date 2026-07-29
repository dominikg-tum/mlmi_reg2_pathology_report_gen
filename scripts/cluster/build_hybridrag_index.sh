#!/bin/bash
#SBATCH --job-name=pathology-build-hybridrag
#SBATCH --chdir=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=NONE
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/logs/build_hybridrag_%j.out
#SBATCH --error=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/logs/build_hybridrag_%j.err
#
# Build HybridRAG Chroma + BM25 index from train-split labels xlsx + reference chunks.
# Split filter uses cases.csv (via wsi_name_map), not xlsx row RNG.
#
# Usage:
#   FORCE_REBUILD=1 sbatch --export=NONE,FORCE_REBUILD=1 \
#     scripts/cluster/build_hybridrag_index.sh
# Coordinate before FORCE_REBUILD on the shared chroma path (Nick).

set -euo pipefail

export PATH="/usr/local/bin:/usr/bin:/bin${PATH:+:${PATH}}"

PINNED_REPO="${MLMI_PINNED_REPO:-/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen}"
export MLMI_PINNED_REPO="${PINNED_REPO}"

SPLIT="${SPLIT:-train}"
FORCE_REBUILD="${FORCE_REBUILD:-}"

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
$(cluster_hybridrag_pip_snippet)
    ARGS=(python -m scripts.memory.build_hybridrag_index --split '${SPLIT}')
    [[ -n '${FORCE_REBUILD}' ]] && ARGS+=(--force-rebuild)
    \"\${ARGS[@]}\"
  "
