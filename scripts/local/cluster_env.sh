# Source from scripts/local/*.sh on your laptop (or any machine with SSH to head).
#
# Override via environment or copy to cluster_env.local.sh (gitignored pattern: *.local.sh).

CLUSTER_SSH_HOST="${CLUSTER_SSH_HOST:-dominikgarstenauer@head.garching.camp.cluster}"
PINNED_REPO="${PINNED_REPO:-/mnt/projects/mlmi/reg2/dominik/repos/mlmi_reg2_pathology_report_gen}"
SHARED_REPO="${SHARED_REPO:-/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen}"
REMOTE_LOGS_DIR="${REMOTE_LOGS_DIR:-/mnt/projects/mlmi/reg2/dominik/logs}"
REMOTE_CACHE_DIR="${REMOTE_CACHE_DIR:-/mnt/projects/mlmi/reg2/dominik/cache}"
REMOTE_LOCK_FILE="${REMOTE_LOCK_FILE:-/mnt/projects/mlmi/reg2/dominik/locks/wsi_batch_local.lock}"

# students_opportunistic: MaxJobs=2 total per user (includes cursor-ssh, wsi-offline, etc.).
MAX_USER_JOBS="${MAX_USER_JOBS:-2}"
# Cap concurrent wsi-offline GPU jobs (usually same as MAX_USER_JOBS minus headroom).
MAX_WSI_JOBS="${MAX_WSI_JOBS:-2}"
POLL_SEC="${POLL_SEC:-30}"

_cluster_ssh() {
  local attempt rc=255
  for attempt in 1 2 3; do
    if ssh -o BatchMode=yes -o ForwardX11=no -o ConnectTimeout=30 \
      "${CLUSTER_SSH_HOST}" "$@"; then
      return 0
    fi
    rc=$?
    [[ "${attempt}" -lt 3 ]] && sleep 3
  done
  return "${rc}"
}

# Parse wsi-index from squeue %i field (e.g. 10709_120 → 120).
_wsi_index_from_squeue_id() {
  local job_id="$1"
  if [[ "${job_id}" == *_* ]]; then
    echo "${job_id##*_}"
  else
    echo "?"
  fi
}

# Local repo root (directory containing scripts/local/)
if [[ -z "${LOCAL_REPO_ROOT:-}" ]]; then
  LOCAL_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

if [[ -f "${LOCAL_REPO_ROOT}/scripts/local/cluster_env.local.sh" ]]; then
  # shellcheck source=/dev/null
  source "${LOCAL_REPO_ROOT}/scripts/local/cluster_env.local.sh"
fi
