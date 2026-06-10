#!/bin/bash
#SBATCH --job-name=pathology-phase2-report
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/phase2_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/phase2_%j.err

set -euo pipefail

SLIDE_ID="${1:-}"
if [[ -z "${SLIDE_ID}" ]]; then
  echo "Usage: sbatch run_phase2.sh CASE.svs" >&2
  exit 1
fi

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
$(cluster_titan_pip_snippet)
$(cluster_hf_login_snippet)
    python -m scripts.inference.run_phase2 --slide-id '${SLIDE_ID}'
  "
