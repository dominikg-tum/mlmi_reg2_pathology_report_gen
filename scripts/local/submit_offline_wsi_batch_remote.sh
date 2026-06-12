#!/bin/bash
# Batch offline WSI submitter — runs on your laptop, SLURM on cluster, code from pinned repo.
#
# SAFETY: Refuses to start if the cluster-side submit_offline_wsi_batch.sh is still running,
# unless you pass --handoff (see scripts/local/README.md).
#
# Usage:
#   bash scripts/local/submit_offline_wsi_batch_remote.sh --start 117 --end 459
#   bash scripts/local/submit_offline_wsi_batch_remote.sh --start 117 --end 459 --handoff
#   bash scripts/local/auto_handoff_wsi_batch.sh   # recommended: auto-detect + handoff

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${SCRIPT_DIR}/cluster_env.sh"
# shellcheck source=handoff_lib.sh
source "${SCRIPT_DIR}/handoff_lib.sh"

START=0
END=459
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

wait_for_wsi_slot() {
  while true; do
    local total wsi_running
    read -r total wsi_running < <(
      _cluster_ssh "printf '%s %s' \"\$(squeue -u \\\"\\\$(whoami)\\\" -h 2>/dev/null | wc -l)\" \"\$(squeue -u \\\"\\\$(whoami)\\\" -n wsi-offline-pipeline -h 2>/dev/null | wc -l)\""
    )
    total=${total// /}
    wsi_running=${wsi_running// /}
    if ((total < MAX_USER_JOBS && wsi_running < MAX_WSI_JOBS)); then
      return 0
    fi
    echo "Waiting for job slot (total ${total}/${MAX_USER_JOBS}, wsi-offline ${wsi_running}/${MAX_WSI_JOBS})..."
    sleep "${POLL_SEC}"
  done
}

fetch_incomplete_indices() {
  local start="$1" end="$2"
  remote_ensure_wsi_manifest "${PINNED_REPO}"
  _cluster_ssh "python3 -u '${PINNED_REPO}/scripts/local/remote_cache_check.py' '${PINNED_REPO}' incomplete '${start}' '${end}'"
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

echo "Scanning wsi-index ${START}-${END} for incomplete slides (one fast pass on head)..."
mapfile -t todo < <(fetch_incomplete_indices "${START}" "${END}")

if ((${#todo[@]} == 0)); then
  echo "All slides complete in range ${START}-${END}."
  exit 0
fi

echo "Found ${#todo[@]} incomplete slide(s) to submit."

submitted=0
skipped=0
for idx in "${todo[@]}"; do
  if remote_slide_complete "${idx}"; then
    echo "SKIP wsi-index ${idx} (artifacts complete since scan)"
    skipped=$((skipped + 1))
    continue
  fi

  wait_for_wsi_slot
  if ((DO_SYNC)); then
    bash "${SCRIPT_DIR}/sync_repo_to_cluster.sh"
  fi

  echo "Submitting wsi-index ${idx}..."
  out=$(_cluster_ssh "cd '${PINNED_REPO}' && sbatch --export=ALL,MLMI_PINNED_REPO='${PINNED_REPO}' --array='${idx}' scripts/cluster/run_offline_wsi_pinned.sh")
  echo "${out} (wsi-index ${idx}, pinned repo)"
  submitted=$((submitted + 1))
done

echo "Done: submitted ${submitted}, skipped ${skipped} since scan (wsi-index ${START}-${END}, pinned repo)."
