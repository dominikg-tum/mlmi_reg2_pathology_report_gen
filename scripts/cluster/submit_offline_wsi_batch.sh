#!/bin/bash
# Submit offline WSI jobs on the cluster head (no laptop / VPN after start).
# Respects MaxJobs=2; skips slides that already have patch_embeddings_20x.pt.
#
# Usage (on head.garching.camp.cluster):
#   bash scripts/cluster/submit_offline_wsi_batch.sh
#   bash scripts/cluster/submit_offline_wsi_batch.sh --start 3 --end 463
#
# Detached (VPN-safe):
#   bash scripts/cluster/start_offline_wsi_batch_daemon.sh --start 3

set -euo pipefail

START=0
END=463
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

PINNED_REPO="${MLMI_PINNED_REPO:-/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen}"
CACHE_ROOT="${MLMI_CACHE_ROOT:-/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/cache_20x_v2}"
NAME_MAP="${PINNED_REPO}/data/manifests/wsi_name_map.csv"

# shellcheck source=load_paths.sh
source "${PINNED_REPO}/scripts/cluster/load_paths.sh"
load_cluster_paths "${PINNED_REPO}/configs/paths.yaml"

slide_artifacts_complete() {
  local idx="$1"
  local slide_id
  # Avoid Python on the head node (can hang under load); CSV wsi_index -> slide_id.
  slide_id=$(
    awk -F, -v i="${idx}" 'NR > 1 && $1 == i { gsub(/ /, "", $3); print $3; exit }' "${NAME_MAP}"
  )
  if [[ -z "${slide_id}" ]]; then
    return 1
  fi
  [[ -f "${CACHE_ROOT}/${slide_id}/patch_embeddings_20x.pt" ]]
}

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
echo "Batch submitter: wsi-index ${START}-${END} cache=${CACHE_ROOT}" >&2
for idx in $(seq "${START}" "${END}"); do
  if slide_artifacts_complete "${idx}"; then
    echo "SKIP wsi-index ${idx} (patch_embeddings_20x.pt exists)"
    skipped=$((skipped + 1))
    continue
  fi

  wait_for_slot
  out=$(
    cd "${PINNED_REPO}" && sbatch \
      --export=NONE,MLMI_PINNED_REPO="${PINNED_REPO}" \
      --array="${idx}" scripts/cluster/run_offline_wsi_pinned.sh
  )
  echo "${out} (wsi-index ${idx})"
  submitted=$((submitted + 1))
done

echo "Done: submitted ${submitted}, skipped ${skipped} (wsi-index ${START}-${END})."
