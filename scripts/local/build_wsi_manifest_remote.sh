#!/bin/bash
# One-time (or refresh) wsi-index manifest on cluster — fast via thumbnail bank.
#
# Usage:
#   bash scripts/local/build_wsi_manifest_remote.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${SCRIPT_DIR}/cluster_env.sh"
# shellcheck source=handoff_lib.sh
source "${SCRIPT_DIR}/handoff_lib.sh"

bash "${SCRIPT_DIR}/sync_repo_to_cluster.sh"
remote_ensure_wsi_manifest "${PINNED_REPO}"
echo "Manifest ready under ${PINNED_REPO} (see cache_root/wsi_svs_index_manifest.txt on cluster)."
