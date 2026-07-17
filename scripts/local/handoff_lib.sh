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

# Resume index = count of patch_embeddings_20x.pt under cache (fast find).
remote_fast_resume_index() {
  local n
  echo "Resume index from patch_embeddings_20x.pt count in ${REMOTE_CACHE_DIR}..." >&2
  n=$(_cluster_ssh "find '${REMOTE_CACHE_DIR}' -maxdepth 2 -name patch_embeddings_20x.pt 2>/dev/null | wc -l")
  echo "${n// /}"
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
