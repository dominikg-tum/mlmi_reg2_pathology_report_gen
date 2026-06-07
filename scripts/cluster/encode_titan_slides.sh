#!/bin/bash
#SBATCH --job-name=titan-slide
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/titan_slide_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/titan_slide_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_paths.sh
source "${SCRIPT_DIR}/load_paths.sh"
load_cluster_paths

mkdir -p "${LOGS_DIR}"

LIMIT="${LIMIT:-0}"
SLIDE="${SLIDE:-}"
MAX_PATCHES="${MAX_PATCHES:-512}"

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    pip install -q openslide-python pillow pyyaml tqdm transformers torch 2>/dev/null || true
    # HF token for MahmoodLab/TITAN — set once: export HF_TOKEN=hf_...
    if [[ -n \"\${HF_TOKEN:-}\" ]]; then
      huggingface-cli login --token \"\${HF_TOKEN}\" 2>/dev/null || true
    fi
    ARGS=(python -m scripts.vision.encode_slide_embeddings --max-patches '${MAX_PATCHES}')
    [[ -n '${SLIDE}' ]] && ARGS+=(--slide '${SLIDE}')
    [[ '${LIMIT}' != '0' ]] && ARGS+=(--limit '${LIMIT}')
    \"\${ARGS[@]}\"
  "
