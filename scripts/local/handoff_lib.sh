# Shared helpers for cluster → pinned-repo handoff (sourced, not executed).

stop_cluster_batch_submitter() {
  if _cluster_ssh "pgrep -f 'bash scripts/cluster/submit_offline_wsi_batch.sh' >/dev/null 2>&1"; then
    echo "Stopping cluster submit_offline_wsi_batch.sh (prevents next slide on shared repo)..."
    _cluster_ssh "pkill -f 'bash scripts/cluster/submit_offline_wsi_batch.sh' || true"
    sleep 2
    if _cluster_ssh "pgrep -f 'bash scripts/cluster/submit_offline_wsi_batch.sh' >/dev/null 2>&1"; then
      echo "WARNING: cluster batch submitter still running; retrying pkill -9" >&2
      _cluster_ssh "pkill -9 -f 'bash scripts/cluster/submit_offline_wsi_batch.sh' || true"
      sleep 1
    fi
  else
    echo "No cluster batch submitter running."
  fi
}

# Prints one line: "job_id array_task" per running wsi-offline job, or nothing.
list_running_wsi_jobs() {
  _cluster_ssh "squeue -u \"\$(whoami)\" -n wsi-offline-pipeline -h -o '%i %a' 2>/dev/null" || true
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
    job_id="${line%% *}"
    task="${line##* }"
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

# First wsi-index without full offline artifacts (uses NFS cache; repo only for imports).
remote_first_incomplete_index() {
  local repo="${1:-${PINNED_REPO}}"
  _cluster_ssh "python3 - '${repo}'" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo))
from scripts.vision._common import default_cache_root, default_data_dir, load_vision_config
from vision.cache import slide_cache_dir
from vision.wsi_io import resolve_wsi_files, slide_id_from_path

vcfg = load_vision_config()
data_dir = default_data_dir()
cache_root = default_cache_root(vcfg)
required = [
    "patch_embeddings_10x.pt",
    "patch_embeddings_20x.pt",
    "kmeans_centroids_10x.pt",
    "kmeans_centroids_20x.pt",
    "slide_embedding.pt",
]
for idx in range(460):
    try:
        svs = resolve_wsi_files(data_dir, wsi_index=idx)[0]
        out_dir = slide_cache_dir(cache_root, slide_id_from_path(svs))
    except Exception:
        print(idx)
        raise SystemExit(0)
    if not all((out_dir / n).exists() for n in required):
        print(idx)
        raise SystemExit(0)
print(460)
PY
}
