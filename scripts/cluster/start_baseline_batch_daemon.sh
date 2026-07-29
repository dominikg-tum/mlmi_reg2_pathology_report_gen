#!/bin/bash
# Start submit_baseline_batch.sh detached on the cluster head (survives SSH/VPN disconnect).
#
# Run once while on VPN:
#   ssh dominikgarstenauer@head.garching.camp.cluster
#   cd /mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen
#   bash scripts/cluster/start_baseline_batch_daemon.sh --baseline a
#
# Monitor:
#   tail -f /mnt/projects/mlmi/TUMUntera/dominik_garstenauer/logs/baseline_batch_submitter.log
#   squeue -u dominikgarstenauer -n path-baseline-a-test
#   # job name is path-baseline-${BASELINE}-${SPLIT}
set -euo pipefail

PINNED="${MLMI_PINNED_REPO:-/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen}"
LOG_DIR="/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/logs"
LOG="${LOG_DIR}/baseline_batch_submitter.log"
LOCK="/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/locks/baseline_batch_head.lock"

mkdir -p "${LOG_DIR}" "$(dirname "${LOCK}")"

if [[ -f "${LOCK}" ]] && kill -0 "$(cat "${LOCK}")" 2>/dev/null; then
  echo "Batch submitter already running (pid $(cat "${LOCK}")). Log: ${LOG}" >&2
  exit 1
fi

cd "${PINNED}"
nohup stdbuf -oL -eL bash scripts/cluster/submit_baseline_batch.sh "$@" >> "${LOG}" 2>&1 &
echo $! > "${LOCK}"
echo "Started batch submitter pid=$(cat "${LOCK}")"
echo "Log: ${LOG}"
echo "Disconnect VPN safely; jobs + submitter keep running on head."
