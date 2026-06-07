#!/bin/bash
#SBATCH --job-name=wsi-tile
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-219
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/tile_%A_%a.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/tile_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_paths.sh
source "${SCRIPT_DIR}/load_paths.sh"
load_cluster_paths

mkdir -p "${LOGS_DIR}"

CONTAINER="${PERSONAL_CONTAINER:-/mnt/projects/mlmi/reg2/containers/dominik_20260529_base.sqsh}"

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    pip install -q openslide-python pillow pyyaml tqdm 2>/dev/null || true
    python -m scripts.vision.tile_slides \
      --wsi-index '${SLURM_ARRAY_TASK_ID}' \
      --level medium \
      --level high
  "
