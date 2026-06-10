#!/bin/bash
#SBATCH --job-name=wsi-levels
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/wsi_levels_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/wsi_levels_%j.err

# Inspect one .svs pyramid (level_downsamples, mpp-x, ×4 fallback).
# Safe: read-only metadata, no GPU, short CPU job on a compute node.
#
# Usage (from cluster head — never run Python openslide on the head node):
#   sbatch scripts/cluster/verify_wsi_levels.sh
#   SLIDE='case01.svs' sbatch scripts/cluster/verify_wsi_levels.sh
#   DATA_DIR=/path/to/slides SLIDE='case01.svs' sbatch scripts/cluster/verify_wsi_levels.sh
#   tail -f /mnt/projects/mlmi/reg2/dominik/logs/wsi_levels_<JOBID>.out
#
# Interactive (compute node):
#   srun --partition=24g --qos=students_normal --gres=gpu:1 --pty bash -l
#   SLIDE='case01.svs' bash scripts/cluster/verify_wsi_levels.sh

set -euo pipefail

# shellcheck source=load_paths.sh
source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths

mkdir -p "${LOGS_DIR}"

SLIDE="${SLIDE:-}"
DATA_DIR="${DATA_DIR:-}"

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    pip install -q openslide-python pillow pyyaml pandas openpyxl 2>/dev/null || true
    ARGS=(python -m scripts.vision.verify_wsi_levels)
    [[ -n '${DATA_DIR}' ]] && ARGS+=(--data-dir '${DATA_DIR}')
    [[ -n '${SLIDE}' ]] && ARGS+=(--slide '${SLIDE}')
    \"\${ARGS[@]}\"
  "
