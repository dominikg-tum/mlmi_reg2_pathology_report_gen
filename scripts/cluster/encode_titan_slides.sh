#!/bin/bash
#SBATCH --job-name=titan-slide
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/titan_slide_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/titan_slide_%j.err

set -euo pipefail

# shellcheck source=load_paths.sh
source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths

mkdir -p "${LOGS_DIR}"

LIMIT="${LIMIT:-0}"
SLIDE="${SLIDE:-}"
MAX_PATCHES="${MAX_PATCHES:-512}"
readarray -t _ENROOT_HF_ENV < <(cluster_enroot_hf_env)

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${_ENROOT_HF_ENV[@]}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
$(cluster_titan_pip_snippet)
$(cluster_hf_login_snippet)
    ARGS=(python -m scripts.vision.encode_slide_embeddings --max-patches '${MAX_PATCHES}')
    [[ -n '${SLIDE}' ]] && ARGS+=(--slide '${SLIDE}')
    [[ '${LIMIT}' != '0' ]] && ARGS+=(--limit '${LIMIT}')
    \"\${ARGS[@]}\"
  "
