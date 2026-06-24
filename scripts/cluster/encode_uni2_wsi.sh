#!/bin/bash
#SBATCH --job-name=uni2-wsi
#SBATCH --chdir=/mnt/projects/mlmi/reg2/dominik/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/uni2_wsi_%A_%a.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/uni2_wsi_%A_%a.err

set -euo pipefail

LEVEL_ARGS=()
for level in 1.25x 2.5x 5x 10x; do
  LEVEL_ARGS+=(--level "${level}")
done

EXTRA_ARGS=("$@")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  EXTRA_ARGS+=(--wsi-index "${SLURM_ARRAY_TASK_ID}")
fi

UNI_REPO_PATH="${UNI_REPO_PATH:-/mnt/projects/mlmi/reg2/repos/UNI}"

python -m scripts.vision.encode_uni2_wsi \
  "${LEVEL_ARGS[@]}" \
  --repo-path "${UNI_REPO_PATH}" \
  "${EXTRA_ARGS[@]}"
