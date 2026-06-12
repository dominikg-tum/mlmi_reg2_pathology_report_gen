# Source from scripts/local/*.sh on your laptop (or any machine with SSH to head).
#
# Override via environment or copy to cluster_env.local.sh (gitignored pattern: *.local.sh).

CLUSTER_SSH_HOST="${CLUSTER_SSH_HOST:-dominikgarstenauer@head.garching.camp.cluster}"
PINNED_REPO="${PINNED_REPO:-/mnt/projects/mlmi/reg2/dominik/repos/mlmi_reg2_pathology_report_gen}"
SHARED_REPO="${SHARED_REPO:-/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen}"
REMOTE_LOGS_DIR="${REMOTE_LOGS_DIR:-/mnt/projects/mlmi/reg2/dominik/logs}"
REMOTE_LOCK_FILE="${REMOTE_LOCK_FILE:-/mnt/projects/mlmi/reg2/dominik/locks/wsi_batch_local.lock}"

# Match students_opportunistic: up to 2 concurrent wsi-offline GPU jobs (cursor-ssh is CPU-only).
MAX_WSI_JOBS="${MAX_WSI_JOBS:-2}"
POLL_SEC="${POLL_SEC:-30}"

_cluster_ssh() {
  ssh -o BatchMode=yes "${CLUSTER_SSH_HOST}" "$@"
}

# Local repo root (directory containing scripts/local/)
if [[ -z "${LOCAL_REPO_ROOT:-}" ]]; then
  LOCAL_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

if [[ -f "${LOCAL_REPO_ROOT}/scripts/local/cluster_env.local.sh" ]]; then
  # shellcheck source=/dev/null
  source "${LOCAL_REPO_ROOT}/scripts/local/cluster_env.local.sh"
fi
