# Shared helpers for cluster → pinned-repo handoff (sourced, not executed).

stop_cluster_batch_submitter() {
  if _cluster_ssh "pgrep -f 'bash scripts/cluster/submit_offline_wsi_batch.sh' >/dev/null 2>&1"; then
    echo "Stopping cluster submit_offline_wsi_batch.sh (prevents next slide on shared repo)..."
    _cluster_ssh "pkill -f 'bash scripts/cluster/submit_offline_wsi_batch.sh' || true" || true
    sleep 2
    if _cluster_ssh "pgrep -f 'bash scripts/cluster/submit_offline_wsi_batch.sh' >/dev/null 2>&1"; then
      echo "WARNING: cluster batch submitter still running; retrying pkill -9" >&2
      _cluster_ssh "pkill -9 -f 'bash scripts/cluster/submit_offline_wsi_batch.sh' || true" || true
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

# First wsi-index without full offline artifacts (single .svs listing on head).
remote_first_incomplete_index() {
  local repo="${1:-${PINNED_REPO}}"
  local out rc
  echo "Scanning cache for first incomplete wsi-index (one .svs listing on head)..." >&2
  if ! out=$(_cluster_ssh "python3 - '${repo}'" 2>&1 <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo))
from scripts.vision._common import default_cache_root, default_data_dir, load_vision_config
from vision.cache import slide_cache_dir
from vision.wsi_io import find_svs_files, slide_id_from_path

vcfg = load_vision_config()
data_dir = default_data_dir()
cache_root = default_cache_root(vcfg)
required = (
    "patch_embeddings_10x.pt",
    "patch_embeddings_20x.pt",
    "kmeans_centroids_10x.pt",
    "kmeans_centroids_20x.pt",
    "slide_embedding.pt",
)
files = find_svs_files(data_dir)
for idx, svs in enumerate(files):
    out_dir = slide_cache_dir(cache_root, slide_id_from_path(svs))
    if not all((out_dir / name).exists() for name in required):
        print(idx)
        raise SystemExit(0)
print(len(files))
PY
  ); then
    echo "ERROR: remote_first_incomplete_index failed on head:" >&2
    echo "${out}" >&2
    return 1
  fi
  echo "${out}" | tail -1 | tr -d '[:space:]'
}

# True if slide at wsi-index has all required cache artifacts (one .svs listing per call).
remote_slide_complete() {
  local idx="$1"
  local repo="${2:-${PINNED_REPO}}"
  _cluster_ssh "python3 - '${repo}' '${idx}'" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1])
idx = int(sys.argv[2])
sys.path.insert(0, str(repo))
from scripts.vision._common import default_cache_root, default_data_dir, load_vision_config
from vision.cache import slide_cache_dir
from vision.wsi_io import find_svs_files, slide_id_from_path

vcfg = load_vision_config()
data_dir = default_data_dir()
cache_root = default_cache_root(vcfg)
files = find_svs_files(data_dir)
if idx < 0 or idx >= len(files):
    raise SystemExit(1)
out_dir = slide_cache_dir(cache_root, slide_id_from_path(files[idx]))
required = (
    "patch_embeddings_10x.pt",
    "patch_embeddings_20x.pt",
    "kmeans_centroids_10x.pt",
    "kmeans_centroids_20x.pt",
    "slide_embedding.pt",
)
if all((out_dir / name).exists() for name in required):
    raise SystemExit(0)
raise SystemExit(1)
PY
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
