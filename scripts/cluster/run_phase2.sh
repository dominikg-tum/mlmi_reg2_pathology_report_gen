#!/bin/bash
#SBATCH --job-name=pathology-phase2-report
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/phase2_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/phase2_%j.err

set -euo pipefail

SLIDE_ID="${1:-}"
if [[ -z "${SLIDE_ID}" ]]; then
  echo "Usage: sbatch run_phase2.sh CASE.svs" >&2
  exit 1
fi

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
    pip install -q pyyaml transformers torch huggingface_hub 2>/dev/null || true
    python -m scripts.inference.run_phase2 --slide-id '${SLIDE_ID}'
  "
