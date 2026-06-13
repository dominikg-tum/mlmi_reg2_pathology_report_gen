#!/bin/bash
# Rsync this repo (laptop feature branch) → pinned path on cluster NFS.
#
# Usage:
#   bash scripts/local/sync_repo_to_cluster.sh
#   bash scripts/local/sync_repo_to_cluster.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${SCRIPT_DIR}/cluster_env.sh"

DRY_RUN=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=(--dry-run); shift ;;
    -h | --help)
      echo "Usage: $0 [--dry-run]"
      echo "Sync ${LOCAL_REPO_ROOT} → ${CLUSTER_SSH_HOST}:${PINNED_REPO}"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

echo "Syncing → ${CLUSTER_SSH_HOST}:${PINNED_REPO}"
_cluster_ssh "mkdir -p '${PINNED_REPO}'"

rsync -avz -e "ssh -o ForwardX11=no" "${DRY_RUN[@]}" \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.venv/' \
  --exclude '.mypy_cache/' \
  --exclude '*.pyc' \
  --exclude 'ssh.out' \
  --exclude 'ssh.err' \
  "${LOCAL_REPO_ROOT}/" "${CLUSTER_SSH_HOST}:${PINNED_REPO}/"

_cluster_ssh "chmod +x '${PINNED_REPO}/scripts/cluster/'*.sh '${PINNED_REPO}/scripts/local/'*.sh 2>/dev/null || true"
echo "Done."
