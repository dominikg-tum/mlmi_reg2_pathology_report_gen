#!/bin/bash
#SBATCH --job-name=wsi-offline-pipeline
#SBATCH --chdir=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen
# Avoid --export=ALL: on requeue SLURM re-runs login-shell env retrieval and can
# hold the job as "user env retrieval failed requeued held". Job sets its own env.
#SBATCH --export=NONE
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-463
#SBATCH --output=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/logs/offline_%A_%a.out
#SBATCH --error=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/logs/offline_%A_%a.err
#
# Runs code from user.pinned_repo_dir (laptop rsync target under TUMUntera).
# Submit via scripts/local/submit_offline_wsi_remote.sh — not the team shared repo checkout.

set -euo pipefail

# Minimal PATH for enroot / python3 when submitted with --export=NONE.
export PATH="/usr/local/bin:/usr/bin:/bin${PATH:+:${PATH}}"

PINNED_REPO="${MLMI_PINNED_REPO:-/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen}"
export MLMI_PINNED_REPO="${PINNED_REPO}"

# shellcheck source=load_paths.sh
source "${PINNED_REPO}/scripts/cluster/load_paths.sh"
load_cluster_paths "${PINNED_REPO}/configs/paths.yaml"
REPO="${PINNED_REPO}"

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
