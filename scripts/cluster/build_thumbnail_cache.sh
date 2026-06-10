#!/bin/bash
#SBATCH --job-name=thumb-cache
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/thumb_cache_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/thumb_cache_%j.err

set -euo pipefail

# shellcheck source=load_paths.sh
source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths

mkdir -p "${LOGS_DIR}"

LIMIT="${LIMIT:-0}"
SLIDE="${SLIDE:-}"

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    pip install -q openslide-python pillow pyyaml tqdm 2>/dev/null || true
    ARGS=(python -m scripts.vision.build_thumbnail_cache)
    [[ -n '${SLIDE}' ]] && ARGS+=(--slide '${SLIDE}')
    [[ '${LIMIT}' != '0' ]] && ARGS+=(--limit '${LIMIT}')
    \"\${ARGS[@]}\"
  "
