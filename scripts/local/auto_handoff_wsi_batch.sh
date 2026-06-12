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
#   --start N        Override resume index (default: count of slide_embedding.pt in cache)
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
START=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2 ;;
    --end) END="$2"; shift 2 ;;
    --no-sync) DO_SYNC=0; shift ;;
    -h | --help)
      cat <<'EOF'
Usage: auto_handoff_wsi_batch.sh [--start N] [--end N] [--no-sync]

Watches the cluster, stops the shared-repo batch submitter, waits for the
current wsi-offline job to exit, then runs submit_offline_wsi_batch_remote.sh
from your laptop (pinned code on NFS).

Resume index defaults to slide_embedding.pt count in cache (fast; no manifest).
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

acquire_local_handoff_lock

mapfile -t _initial < <(list_running_wsi_jobs)
if ((${#_initial[@]} > 0)) && [[ -n "${_initial[0]// /}" ]]; then
  echo "Detected running job(s):"
  for line in "${_initial[@]}"; do
    [[ -z "${line// /}" ]] && continue
    job_id="${line// /}"
    echo "  job ${job_id} (wsi-index $(_wsi_index_from_squeue_id "${job_id}"))"
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

if [[ -z "${START}" ]]; then
  if ! START=$(remote_fast_resume_index); then
    echo "ERROR: could not determine resume wsi-index" >&2
    exit 1
  fi
fi
if [[ -z "${START}" ]] || ! [[ "${START}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: invalid wsi-index: '${START}'" >&2
  exit 1
fi
echo "Resume wsi-index: ${START}" >&2

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
