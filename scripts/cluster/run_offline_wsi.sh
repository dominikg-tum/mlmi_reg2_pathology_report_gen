#!/bin/bash
#SBATCH --job-name=wsi-offline-pipeline
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-463
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/offline_%A_%a.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/offline_%A_%a.err

set -euo pipefail

# shellcheck source=load_paths.sh
source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths

mkdir -p "${LOGS_DIR}"

CONTAINER="${PERSONAL_CONTAINER:-/mnt/projects/mlmi/reg2/containers/dominik_20260529_base.sqsh}"
readarray -t _ENROOT_HF_ENV < <(cluster_enroot_hf_env)

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${_ENROOT_HF_ENV[@]}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
$(cluster_offline_pip_snippet)
$(cluster_hf_login_snippet)
    python -m scripts.preprocess.run_offline_wsi \
      --wsi-index '${SLURM_ARRAY_TASK_ID}'
  "
