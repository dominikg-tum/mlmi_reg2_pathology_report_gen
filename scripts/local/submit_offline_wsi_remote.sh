#!/bin/bash
# Submit one offline WSI SLURM job from your laptop (pinned repo on cluster).
#
# Usage:
#   bash scripts/local/submit_offline_wsi_remote.sh --wsi-index 117
#   bash scripts/local/submit_offline_wsi_remote.sh --wsi-index 117 --no-sync

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${SCRIPT_DIR}/cluster_env.sh"

WSI_INDEX=""
DO_SYNC=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wsi-index) WSI_INDEX="$2"; shift 2 ;;
    --no-sync) DO_SYNC=0; shift ;;
    -h | --help)
      echo "Usage: $0 --wsi-index N [--no-sync]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${WSI_INDEX}" ]]; then
  echo "Provide --wsi-index N" >&2
  exit 2
fi

if ((DO_SYNC)); then
  bash "${SCRIPT_DIR}/sync_repo_to_cluster.sh"
fi

echo "Submitting wsi-index ${WSI_INDEX} from pinned repo (not team shared checkout)..."
out=$(_cluster_ssh "cd '${PINNED_REPO}' && sbatch --export=NONE,MLMI_PINNED_REPO='${PINNED_REPO}' --array='${WSI_INDEX}' scripts/cluster/run_offline_wsi_pinned.sh")
echo "${out}"
