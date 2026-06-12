# Shared helpers for cluster → pinned-repo handoff (sourced, not executed).

# pgrep pattern: [s]… avoids matching the pgrep process itself (classic footgun).
CLUSTER_BATCH_SUBMITTER_PGREP='[s]cripts/cluster/submit_offline_wsi_batch.sh'

stop_cluster_batch_submitter() {
  if _cluster_ssh "pgrep -f '${CLUSTER_BATCH_SUBMITTER_PGREP}' >/dev/null 2>&1"; then
    echo "Stopping cluster submit_offline_wsi_batch.sh (prevents next slide on shared repo)..."
    _cluster_ssh "pkill -f '${CLUSTER_BATCH_SUBMITTER_PGREP}' || true" || true
    sleep 2
    if _cluster_ssh "pgrep -f '${CLUSTER_BATCH_SUBMITTER_PGREP}' >/dev/null 2>&1"; then
      echo "WARNING: cluster batch submitter still running; retrying pkill -9" >&2
      _cluster_ssh "pkill -9 -f '${CLUSTER_BATCH_SUBMITTER_PGREP}' || true" || true
      sleep 1
    fi
  else
    echo "No cluster batch submitter running."
  fi
}

# Prints one line per running wsi-offline job: SLURM job id (%i, e.g. 10709_120).
list_running_wsi_jobs() {
  _cluster_ssh "squeue -u \"\$(whoami)\" -n wsi-offline-pipeline -h -o '%i' 2>/dev/null" || true
}

wait_for_wsi_jobs_drain() {
  local line job_id task
  mapfile -t _running < <(list_running_wsi_jobs)
  if ((${#_running[@]} == 0)) || [[ -z "${_running[0]// /}" ]]; then
    echo "No wsi-offline-pipeline jobs in queue."
    return 0
  fi

  echo "Waiting for in-flight wsi-offline job(s) to finish (shared-repo code; not interrupted):"
  for line in "${_running[@]}"; do
    [[ -z "${line// /}" ]] && continue
    job_id="${line// /}"
    task=$(_wsi_index_from_squeue_id "${job_id}")
    echo "  job ${job_id} wsi-index ${task}"
  done

  while true; do
    mapfile -t _running < <(list_running_wsi_jobs)
    if ((${#_running[@]} == 0)) || [[ -z "${_running[0]// /}" ]]; then
      echo "All wsi-offline jobs finished."
      return 0
    fi
    sleep "${POLL_SEC}"
  done
}

remote_ensure_wsi_manifest() {
  local repo="${1:-${PINNED_REPO}}"
  echo "Ensuring wsi index manifest on head (thumbnail bank or one-time find)..." >&2
  _cluster_ssh "python3 -u '${repo}/scripts/local/remote_cache_check.py' '${repo}' ensure-manifest -" >&2
}

# First wsi-index without full offline artifacts (uses cached manifest on head).
remote_first_incomplete_index() {
  local repo="${1:-${PINNED_REPO}}"
  local out
  remote_ensure_wsi_manifest "${repo}"
  echo "Scanning cache for first incomplete wsi-index..." >&2
  if ! out=$(_cluster_ssh "python3 -u '${repo}/scripts/local/remote_cache_check.py' '${repo}' first -"); then
    echo "ERROR: remote_first_incomplete_index failed on head" >&2
    return 1
  fi
  echo "${out}" | tail -1 | tr -d '[:space:]'
}

# True if slide at wsi-index has all required cache artifacts.
remote_slide_complete() {
  local idx="$1"
  local repo="${2:-${PINNED_REPO}}"
  _cluster_ssh "python3 -u '${repo}/scripts/local/remote_cache_check.py' '${repo}' check '${idx}'"
}

acquire_local_handoff_lock() {
  local lock="${LOCAL_REPO_ROOT}/.auto_handoff.lock"
  exec 9>"${lock}"
  if ! flock -n 9; then
    echo "ERROR: another auto_handoff is already running (lock: ${lock})" >&2
    echo "  pgrep -af 'auto_handoff_wsi_batch|submit_offline_wsi_batch_remote'" >&2
    exit 1
  fi
  echo "auto_handoff $$ $(date -Iseconds)" >&9
}
