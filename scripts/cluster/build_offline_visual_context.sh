#!/bin/bash
#SBATCH --job-name=visual-context
#SBATCH --chdir=/mnt/projects/mlmi/reg2/backup/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/visual_context_%A_%a.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/visual_context_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIS_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=load_paths.sh
source "${THIS_REPO}/scripts/cluster/load_paths.sh"
load_cluster_paths "${THIS_REPO}/configs/paths.yaml"
REPO="${THIS_REPO}"

mkdir -p "${LOGS_DIR}"

PLIP_LIB_PATH="${PLIP_LIB_PATH:-/mnt/projects/mlmi/reg2/models/plip}"
PLIP_CKPT="${PLIP_CKPT:-/mnt/projects/mlmi/reg2/models/plip}"

PATHAGENT_ROOT="${PATHAGENT_ROOT:-/mnt/projects/mlmi/reg2/backup/PathAgent}"
UNI_REPO_PATH="${UNI_REPO_PATH:-/mnt/projects/mlmi/reg2/repos/UNI}"
UNI2_WEIGHTS_PATH="${UNI2_WEIGHTS_PATH:-/mnt/projects/mlmi/reg2/models/UNI2-h}"
CONTAINER="${PERSONAL_CONTAINER:-/mnt/projects/mlmi/reg2/containers/qwen25_dev_v2.sqsh}"

export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-/tmp/enroot-runtime-${USER}}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-/tmp/enroot-cache-${USER}}"
export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-/tmp/enroot-data-${USER}}"
mkdir -p "${ENROOT_RUNTIME_PATH}" "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}"

EXTRA_ARGS=("$@")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  EXTRA_ARGS+=(--wsi-index "${SLURM_ARRAY_TASK_ID}")
fi

LEVEL_ARGS=()
for level in 1.25x 2.5x 5x 10x; do
  LEVEL_ARGS+=(--level "${level}")
done

readarray -t _ENROOT_HF_ENV < <(cluster_enroot_hf_env)
printf -v LEVEL_ARGS_Q '%q ' "${LEVEL_ARGS[@]}"
printf -v EXTRA_ARGS_Q '%q ' "${EXTRA_ARGS[@]}"

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${_ENROOT_HF_ENV[@]}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
$(cluster_hf_login_snippet)
    pip install -q openslide-python openslide-bin pillow pyyaml tqdm huggingface_hub numpy scipy scikit-learn
    pip install -q timm==0.9.11 transformers==4.51.0 accelerate==1.10.0 qwen-vl-utils==0.0.14 sentencepiece==0.2.1 2>/dev/null || true
    python -m scripts.vision.build_offline_visual_context \
      --pathagent-root '${PATHAGENT_ROOT}' \
      --plip-lib-path '${PLIP_LIB_PATH}' \
      --plip-ckpt '${PLIP_CKPT}' \
      --uni-repo-path '${UNI_REPO_PATH}' \
      --uni2-weights-path '${UNI2_WEIGHTS_PATH}' \
      ${LEVEL_ARGS_Q} \
      ${EXTRA_ARGS_Q}
  "
