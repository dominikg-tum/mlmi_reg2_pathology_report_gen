#!/bin/bash
# Head-node wrapper: submit slide-0 offline smoke test and retry until artifacts exist.
#
# Usage:
#   bash scripts/cluster/retry_smoke_offline.sh
#   bash scripts/cluster/retry_smoke_offline.sh --max-attempts 5 --slide-index 0

set -euo pipefail

MAX_ATTEMPTS=5
SLIDE_INDEX=0
POLL_SEC=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-attempts)
      MAX_ATTEMPTS="$2"
      shift 2
      ;;
    --slide-index)
      SLIDE_INDEX="$2"
      shift 2
      ;;
    -h | --help)
      echo "Usage: $0 [--max-attempts N] [--slide-index I]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# shellcheck source=load_paths.sh
source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths

mkdir -p "${LOGS_DIR}"

readarray -t _RESOLVED < <(
  python3 - "${REPO}" "${SLIDE_INDEX}" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo))
from scripts.vision._common import default_cache_root, load_vision_config
from vision.cache import slide_cache_dir
from vision.wsi_mapping import mapped_slide_ids

vcfg = load_vision_config()
cache_root = default_cache_root(vcfg)
ids = mapped_slide_ids()
idx = int(sys.argv[2])
if idx < 0 or idx >= len(ids):
    raise SystemExit(f"slide_index={idx} out of range for {len(ids)} mapped slides")
sid = ids[idx]
print(sid)
print(slide_cache_dir(cache_root, sid))
PY
)

SLIDE_ID="${_RESOLVED[0]}"
SLIDE_CACHE_DIR="${_RESOLVED[1]}"

# 20x CONCH only — TITAN slide_embedding.pt is optional (Phase 2).
REQUIRED=(
  patch_embeddings_20x.pt
)

clear_failed_markers() {
  rm -f "${LOGS_DIR}/${SLIDE_ID}.encoded_"*.failed
  rm -f "${LOGS_DIR}/${SLIDE_ID}.kmeans_"*.failed
}

artifacts_ok() {
  local missing=()
  for name in "${REQUIRED[@]}"; do
    if [[ ! -f "${SLIDE_CACHE_DIR}/${name}" ]]; then
      missing+=("${name}")
    fi
  done
  if ((${#missing[@]} > 0)); then
    echo "Missing artifacts: ${missing[*]}"
    return 1
  fi
  return 0
}

print_failure_logs() {
  local job_id="$1"
  local out="${LOGS_DIR}/offline_${job_id}_0.out"
  local err="${LOGS_DIR}/offline_${job_id}_0.err"
  echo "=== tail ${out} ==="
  tail -40 "${out}" 2>/dev/null || true
  echo "=== tail ${err} ==="
  tail -40 "${err}" 2>/dev/null || true
  for f in "${LOGS_DIR}/${SLIDE_ID}.encoded_"*.failed "${LOGS_DIR}/${SLIDE_ID}.kmeans_"*.failed; do
    [[ -f "$f" ]] || continue
    echo "=== $(basename "$f") ==="
    head -30 "$f"
  done
}

wait_for_job() {
  local job_id="$1"
  while squeue -j "${job_id}" -h 2>/dev/null | grep -q .; do
    sleep "${POLL_SEC}"
  done
}

attempt=1
last_job=""
while ((attempt <= MAX_ATTEMPTS)); do
  echo ""
  echo "=== Smoke attempt ${attempt}/${MAX_ATTEMPTS} (slide_index=${SLIDE_INDEX}, slide_id=${SLIDE_ID}) ==="
  clear_failed_markers

  submit_out=$(cd "${REPO}" && sbatch --array="${SLIDE_INDEX}" scripts/cluster/run_offline_wsi.sh)
  echo "${submit_out}"
  last_job=$(sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' <<<"${submit_out}")
  if [[ -z "${last_job}" ]]; then
    echo "Failed to parse job id from sbatch output" >&2
    exit 1
  fi

  echo "Waiting for job ${last_job}_0 (poll every ${POLL_SEC}s)..."
  echo "  tail -f ${LOGS_DIR}/offline_${last_job}_0.out"
  wait_for_job "${last_job}"

  if artifacts_ok; then
    echo ""
    echo "SUCCESS on attempt ${attempt} (job ${last_job}_0)"
    ls -la "${SLIDE_CACHE_DIR}/"
    exit 0
  fi

  echo "Attempt ${attempt} failed artifact check (job ${last_job}_0)."
  print_failure_logs "${last_job}"
  attempt=$((attempt + 1))
done

echo ""
echo "FAILED after ${MAX_ATTEMPTS} attempts (last job ${last_job}_0)."
print_failure_logs "${last_job}"
exit 1
