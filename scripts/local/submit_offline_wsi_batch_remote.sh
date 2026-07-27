#!/bin/bash
# Batch offline WSI submitter — runs on your laptop, SLURM on cluster, code from pinned repo.
#
# SAFETY: Refuses to start if the cluster-side submit_offline_wsi_batch.sh is still running,
# unless you pass --handoff (see scripts/local/README.md).
#
# Usage:
#   bash scripts/local/submit_offline_wsi_batch_remote.sh --start 125 --end 459 --handoff
#   bash scripts/local/auto_handoff_wsi_batch.sh   # recommended: auto-detect + handoff

set -euo pipefail

trap 'echo "FATAL: ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${SCRIPT_DIR}/cluster_env.sh"
# shellcheck source=handoff_lib.sh
source "${SCRIPT_DIR}/handoff_lib.sh"

START=0
END=463
HANDOFF=0
DO_SYNC=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2 ;;
    --end) END="$2"; shift 2 ;;
    --handoff) HANDOFF=1; shift ;;
    --no-sync) DO_SYNC=0; shift ;;
    -h | --help)
      echo "Usage: $0 [--start N] [--end N] [--handoff] [--no-sync]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cluster_batch_running() {
  _cluster_ssh "pgrep -f '${CLUSTER_BATCH_SUBMITTER_PGREP}' >/dev/null 2>&1"
}

# Count running SLURM jobs for current user on cluster (stdout: integer).
remote_user_job_count() {
  local n
  n=$(_cluster_ssh 'squeue -u "$(whoami)" -h 2>/dev/null | wc -l')
  echo "${n// /}"
}

remote_wsi_job_count() {
  local n
  n=$(_cluster_ssh 'squeue -u "$(whoami)" -n wsi-offline-pipeline -h 2>/dev/null | wc -l')
  echo "${n// /}"
}

wait_for_wsi_slot() {
  local total wsi_running
  while true; do
    if ! total=$(remote_user_job_count); then
      echo "WARNING: squeue (total jobs) failed; retry in ${POLL_SEC}s..." >&2
      sleep "${POLL_SEC}"
      continue
    fi
    if ! wsi_running=$(remote_wsi_job_count); then
      echo "WARNING: squeue (wsi-offline) failed; retry in ${POLL_SEC}s..." >&2
      sleep "${POLL_SEC}"
      continue
    fi
    if ((total < MAX_USER_JOBS && wsi_running < MAX_WSI_JOBS)); then
      return 0
    fi
    echo "Waiting for job slot (total ${total}/${MAX_USER_JOBS}, wsi-offline ${wsi_running}/${MAX_WSI_JOBS})..."
    sleep "${POLL_SEC}"
  done
}

acquire_local_lock() {
  _cluster_ssh "mkdir -p \"\$(dirname '${REMOTE_LOCK_FILE}')\" && ( set -o noclobber; echo \"\$\$ laptop \$(date -Iseconds)\" > '${REMOTE_LOCK_FILE}' )" \
    || {
      echo "Lock exists: ${REMOTE_LOCK_FILE}" >&2
      echo "Another local batch may be running. Remove lock on cluster if stale." >&2
      exit 1
    }
}

acquire_local_batch_lock() {
  local lock="${LOCAL_REPO_ROOT}/.submit_offline_wsi_batch_remote.lock"
  exec 8>"${lock}"
  if ! flock -n 8; then
    echo "ERROR: another submit_offline_wsi_batch_remote is running (lock: ${lock})" >&2
    exit 1
  fi
  echo "batch_remote $$ $(date -Iseconds)" >&8
}

release_local_lock() {
  _cluster_ssh "rm -f '${REMOTE_LOCK_FILE}'" 2>/dev/null || true
}

trap release_local_lock EXIT

if cluster_batch_running; then
  if (( ! HANDOFF )); then
    echo "ERROR: cluster submit_offline_wsi_batch.sh is still running on head." >&2
    echo "  Let it finish (~few days), OR after the current slide completes:" >&2
    echo "    ssh head \"pkill -f 'bash scripts/cluster/submit_offline_wsi_batch.sh'\"" >&2
    echo "  Then re-run with --handoff" >&2
    exit 1
  fi
  echo "WARNING: --handoff set; cluster batch submitter may still be running — avoid duplicate slides."
fi

if ((DO_SYNC)); then
  bash "${SCRIPT_DIR}/sync_repo_to_cluster.sh"
fi

acquire_local_batch_lock
acquire_local_lock

echo "Submitting wsi-index ${START}-${END} (sequential; no head-node cache scan)."

submitted=0
for idx in $(seq "${START}" "${END}"); do
  wait_for_wsi_slot
  if ((DO_SYNC)); then
    bash "${SCRIPT_DIR}/sync_repo_to_cluster.sh"
  fi

  echo "Submitting wsi-index ${idx}..."
  if ! out=$(_cluster_ssh "cd '${PINNED_REPO}' && sbatch --export=NONE,MLMI_PINNED_REPO='${PINNED_REPO}' --array='${idx}' scripts/cluster/run_offline_wsi_pinned.sh"); then
    echo "ERROR: sbatch failed for wsi-index ${idx}" >&2
    exit 1
  fi
  echo "${out} (wsi-index ${idx}, pinned repo)"
  submitted=$((submitted + 1))
done

echo "Done: submitted ${submitted} job(s) (wsi-index ${START}-${END}, pinned repo)."
