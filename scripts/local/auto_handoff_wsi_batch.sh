#!/bin/bash
# Automatic handoff: let the current wsi-offline SLURM job finish, then take over with pinned repo.
#
# How it avoids duplicate slides:
#   1. Detect running wsi-offline-pipeline job(s) via squeue (array task = wsi-index).
#   2. Kill cluster submit_offline_wsi_batch.sh immediately (so it cannot sbatch N+1).
#   3. Wait until in-flight job(s) complete.
#   4. Rsync laptop → pinned repo, start local batch from first incomplete index.
#
# Run from your laptop (VPN + SSH to head):
#   bash scripts/local/auto_handoff_wsi_batch.sh
#   nohup bash scripts/local/auto_handoff_wsi_batch.sh >> ~/wsi_handoff.log 2>&1 &
#
# Options:
#   --end N          Last wsi-index (default 459)
#   --no-sync        Skip initial rsync (not recommended)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${SCRIPT_DIR}/cluster_env.sh"
# shellcheck source=handoff_lib.sh
source "${SCRIPT_DIR}/handoff_lib.sh"

END=459
DO_SYNC=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --end) END="$2"; shift 2 ;;
    --no-sync) DO_SYNC=0; shift ;;
    -h | --help)
      cat <<'EOF'
Usage: auto_handoff_wsi_batch.sh [--end N] [--no-sync]

Watches the cluster, stops the shared-repo batch submitter, waits for the
current wsi-offline job to exit, then runs submit_offline_wsi_batch_remote.sh
from your laptop (pinned code on NFS).

Safe to start while a job is running. Does not scancel the in-flight job.
EOF
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

echo "=== WSI batch auto-handoff ==="
echo "Cluster SSH: ${CLUSTER_SSH_HOST}"
echo "Pinned repo: ${PINNED_REPO}"
echo ""

mapfile -t _initial < <(list_running_wsi_jobs)
if ((${#_initial[@]} > 0)) && [[ -n "${_initial[0]// /}" ]]; then
  echo "Detected running job(s):"
  for line in "${_initial[@]}"; do
    [[ -z "${line// /}" ]] && continue
    echo "  ${line} (wsi-index ${line##* })"
  done
else
  echo "No wsi-offline job running right now."
fi

stop_cluster_batch_submitter
wait_for_wsi_jobs_drain

if ((DO_SYNC)); then
  bash "${SCRIPT_DIR}/sync_repo_to_cluster.sh"
else
  echo "Skipping sync (--no-sync)."
fi

START=$(remote_first_incomplete_index "${PINNED_REPO}")
if [[ -z "${START}" ]] || ! [[ "${START}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: could not determine first incomplete wsi-index" >&2
  exit 1
fi

if ((START > END)); then
  echo "All slides complete through index ${END}. Nothing to submit."
  exit 0
fi

echo ""
echo "Handoff complete. Starting pinned local batch: wsi-index ${START}-${END}"
exec bash "${SCRIPT_DIR}/submit_offline_wsi_batch_remote.sh" \
  --start "${START}" \
  --end "${END}" \
  --handoff \
  --no-sync
