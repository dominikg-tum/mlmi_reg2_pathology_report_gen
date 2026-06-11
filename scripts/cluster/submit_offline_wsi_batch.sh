#!/bin/bash
# Submit offline WSI jobs within QOS MaxSubmit/MaxJobs limits (students: 2 jobs max).
# Submits one array task at a time, waiting for a free slot before each sbatch.
#
# Usage:
#   bash scripts/cluster/submit_offline_wsi_batch.sh
#   bash scripts/cluster/submit_offline_wsi_batch.sh --start 2 --end 459

set -euo pipefail

START=0
END=459
POLL_SEC=30
MAX_USER_JOBS=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2 ;;
    --end) END="$2"; shift 2 ;;
    -h | --help)
      echo "Usage: $0 [--start N] [--end N]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

# shellcheck source=load_paths.sh
source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths

wait_for_slot() {
  while true; do
    local total wsi_running
    total=$(squeue -u "${USER}" -h 2>/dev/null | wc -l)
    wsi_running=$(squeue -u "${USER}" -n wsi-offline-pipeline -h 2>/dev/null | wc -l)
    if ((total < MAX_USER_JOBS)); then
      return 0
    fi
    echo "Waiting for job slot (${total}/${MAX_USER_JOBS} jobs, ${wsi_running} wsi-offline running)..."
    sleep "${POLL_SEC}"
  done
}

submitted=0
skipped=0
for idx in $(seq "${START}" "${END}"); do
  wait_for_slot
  out=$(cd "${REPO}" && sbatch --array="${idx}" scripts/cluster/run_offline_wsi.sh)
  echo "${out} (wsi-index ${idx})"
  submitted=$((submitted + 1))
done

echo "Submitted ${submitted} jobs for wsi-index ${START}-${END} (skipped ${skipped})."
