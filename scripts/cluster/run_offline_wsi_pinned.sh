#!/bin/bash
#SBATCH --job-name=wsi-offline-pipeline
#SBATCH --chdir=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen
# Avoid --export=ALL: on requeue SLURM re-runs login-shell env retrieval and can
# hold the job as "user env retrieval failed requeued held". Job sets its own env.
#SBATCH --export=NONE
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
# Student QoS: MaxJobsPU=2, MaxSubmitPU=2. Use generic GPU request (fair-share).
# Do not --exclude (forbidden) or hard --nodelist / GPU-type pin (starves the queue).
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

# koblenz (TITAN RTX) previously killed offline jobs with RaisedSignal:53 in 0–2s.
# Refuse fast so the MaxJobs=2 slot frees and the batch submitter can retry elsewhere.
HOST="$(hostname -s 2>/dev/null || hostname)"
if [[ "${HOST}" == "koblenz" ]]; then
  echo "REFUSE node=${HOST}: TITAN RTX historically raises signal 53 for this pipeline."
  echo "Exiting 75 so submitter can retry on another 24g node (no embedding written)."
  exit 75
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi -L 2>/dev/null | grep -qi 'TITAN'; then
    echo "REFUSE GPU on ${HOST}: TITAN device listed by nvidia-smi."
    echo "Exiting 75 so submitter can retry on another 24g node (no embedding written)."
    exit 75
  fi
fi

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
