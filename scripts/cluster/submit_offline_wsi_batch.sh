#!/bin/bash
# Submit offline WSI jobs on the cluster head (no laptop / VPN after start).
# Respects students_opportunistic MaxJobsPU=2 / MaxSubmitPU=2.
# Skips slides that already have patch_embeddings_20x.pt and indices whose .svs
# is missing from DATA_DIR (see data/manifests/wsi_missing_on_disk.md).
# Re-scans until a full pass submits nothing so koblenz refuse-exits can retry.
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
MAX_ROUNDS=50

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
DATA_DIR="${MLMI_DATA_DIR:-/mnt/projects/mlmi/TUMUntera/TUM_Untera_data}"

# shellcheck source=load_paths.sh
source "${PINNED_REPO}/scripts/cluster/load_paths.sh"
load_cluster_paths "${PINNED_REPO}/configs/paths.yaml"

slide_row_fields() {
  local idx="$1"
  # CSV: wsi_index,tum_num,slide_id,disk_name,...
  awk -F, -v i="${idx}" 'NR > 1 && $1 == i {
    gsub(/ /, "", $3); gsub(/ /, "", $4);
    print $3 "\t" $4;
    exit
  }' "${NAME_MAP}"
}

slide_artifacts_complete() {
  local idx="$1"
  local slide_id disk_name
  IFS=$'\t' read -r slide_id disk_name < <(slide_row_fields "${idx}")
  if [[ -z "${slide_id}" ]]; then
    return 1
  fi
  [[ -f "${CACHE_ROOT}/${slide_id}/patch_embeddings_20x.pt" ]]
}

slide_svs_available() {
  local idx="$1"
  local slide_id disk_name
  IFS=$'\t' read -r slide_id disk_name < <(slide_row_fields "${idx}")
  if [[ -z "${slide_id}" ]]; then
    return 1
  fi
  [[ -f "${DATA_DIR}/${slide_id}" || -f "${DATA_DIR}/${disk_name}" ]]
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

submitted_total=0
echo "Batch submitter: wsi-index ${START}-${END} cache=${CACHE_ROOT} data=${DATA_DIR}" >&2

for ((round = 1; round <= MAX_ROUNDS; round++)); do
  submitted_round=0
  skipped_done=0
  skipped_missing_svs=0
  echo "=== pass ${round}/${MAX_ROUNDS} ===" >&2

  for idx in $(seq "${START}" "${END}"); do
    if slide_artifacts_complete "${idx}"; then
      skipped_done=$((skipped_done + 1))
      continue
    fi
    if ! slide_svs_available "${idx}"; then
      skipped_missing_svs=$((skipped_missing_svs + 1))
      continue
    fi

    wait_for_slot
    out=$(
      cd "${PINNED_REPO}" && sbatch \
        --export=NONE,MLMI_PINNED_REPO="${PINNED_REPO}" \
        --array="${idx}" scripts/cluster/run_offline_wsi_pinned.sh
    )
    echo "${out} (wsi-index ${idx}, pass ${round})"
    submitted_round=$((submitted_round + 1))
    submitted_total=$((submitted_total + 1))
  done

  echo "Pass ${round}: submitted ${submitted_round}, already-done ${skipped_done}, missing-svs ${skipped_missing_svs}"
  if ((submitted_round == 0)); then
    echo "Done: no more submitable indices. total_submitted=${submitted_total}"
    exit 0
  fi
  # Let in-flight jobs finish before rescanning (MaxJobs=2).
  wait_for_slot
  sleep "${POLL_SEC}"
done

echo "Stopped after ${MAX_ROUNDS} passes (total_submitted=${submitted_total}). Re-run to continue."
exit 1
