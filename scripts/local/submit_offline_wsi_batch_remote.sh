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
  _cluster_ssh "pgrep -f 'bash scripts/cluster/submit_offline_wsi_batch.sh' >/dev/null 2>&1"
}

remote_slide_complete() {
  local idx="$1"
  _cluster_ssh "python3 - '${PINNED_REPO}' '${idx}'" <<'PY'
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
svs = resolve_wsi_files(data_dir, wsi_index=int(sys.argv[2]))[0]
out_dir = slide_cache_dir(cache_root, slide_id_from_path(svs))
required = [
    "patch_embeddings_10x.pt",
    "patch_embeddings_20x.pt",
    "kmeans_centroids_10x.pt",
    "kmeans_centroids_20x.pt",
    "slide_embedding.pt",
]
missing = [name for name in required if not (out_dir / name).exists()]
if missing:
    raise SystemExit(1)
PY
}

wait_for_wsi_slot() {
  while true; do
    local wsi_running
    wsi_running=$(_cluster_ssh "squeue -u \"\$(whoami)\" -n wsi-offline-pipeline -h 2>/dev/null | wc -l")
    if ((wsi_running < MAX_WSI_JOBS)); then
      return 0
    fi
    echo "Waiting for wsi-offline slot (${wsi_running}/${MAX_WSI_JOBS})..."
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

acquire_local_lock

submitted=0
skipped=0
for idx in $(seq "${START}" "${END}"); do
  if remote_slide_complete "${idx}"; then
    echo "SKIP wsi-index ${idx} (artifacts complete)"
    skipped=$((skipped + 1))
    continue
  fi

  wait_for_wsi_slot
  if ((DO_SYNC)); then
    bash "${SCRIPT_DIR}/sync_repo_to_cluster.sh"
  fi

  out=$(_cluster_ssh "cd '${PINNED_REPO}' && sbatch --export=ALL,MLMI_PINNED_REPO='${PINNED_REPO}' --array='${idx}' scripts/cluster/run_offline_wsi_pinned.sh")
  echo "${out} (wsi-index ${idx}, pinned repo)"
  submitted=$((submitted + 1))
done

echo "Done: submitted ${submitted}, skipped ${skipped} (wsi-index ${START}-${END}, pinned repo)."
