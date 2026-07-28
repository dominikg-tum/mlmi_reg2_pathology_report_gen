#!/bin/bash
# Submit baseline batch jobs one at a time on the cluster head (no laptop / VPN after start).
# Respects the students_opportunistic QOS submit limit (max 1 concurrent job under that QOS).
#
# Usage (on head):
#   bash scripts/cluster/submit_baseline_batch.sh --baseline b2
#   bash scripts/cluster/submit_baseline_batch.sh --baseline b2 --start 3 --end 69
#
# Detached (VPN-safe):
#   bash scripts/cluster/start_baseline_batch_daemon.sh --baseline b2

set -euo pipefail

START=0
END=69
POLL_SEC=30
MAX_OPPORTUNISTIC_JOBS=2
BASELINE="a"
SPLIT="test"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2 ;;
    --end) END="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    -h | --help)
      echo "Usage: $0 [--baseline a|b1|b2] [--split test] [--start N] [--end N]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

# NOTE: adjust these two paths to match your actual personal checkout/folder on the cluster
PINNED_REPO="${MLMI_PINNED_REPO:-/mnt/projects/mlmi/reg2/nick/repos/mlmi_reg2_pathology_report_gen}"
PERSONAL_DIR="${MLMI_PERSONAL_DIR:-/mnt/projects/mlmi/reg2/nick}"

wait_for_slot() {
  while true; do
    local opp_jobs
    opp_jobs=$(squeue -u "${USER}" -h --qos=students_opportunistic 2>/dev/null | wc -l)
    if ((opp_jobs < MAX_OPPORTUNISTIC_JOBS)); then
      return 0
    fi
    echo "Waiting for students_opportunistic slot (${opp_jobs}/${MAX_OPPORTUNISTIC_JOBS})..."
    sleep "${POLL_SEC}"
  done
}

submitted=0
echo "Batch submitter: baseline=${BASELINE} split=${SPLIT} slide-index ${START}-${END}" >&2
for idx in $(seq "${START}" "${END}"); do
  wait_for_slot

  out=$(
    cd "${PINNED_REPO}" && sbatch \
      --export=NONE,BASELINE="${BASELINE}",SPLIT="${SPLIT}" \
      --array="${idx}" scripts/cluster/run_baseline_batch.sh
  )
  echo "${out} (slide-index ${idx})"

  # --- CodeRabbit FIX START ---
  job_id=$(echo "${out}" | awk '{print $4}')
  while ! squeue -j "${job_id}" &>/dev/null; do
      sleep 2
  done
  # --- CodeRabbit FIX END ---

  submitted=$((submitted + 1))
done

echo "Done: submitted ${submitted} tasks (slide-index ${START}-${END})."