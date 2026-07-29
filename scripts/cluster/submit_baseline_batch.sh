#!/bin/bash
# Submit baseline batch jobs one at a time on the cluster head (no laptop / VPN after start).
# Caps concurrent students_opportunistic jobs at MAX_OPPORTUNISTIC_JOBS (default 2).
#
# Usage (on head):
#   bash scripts/cluster/submit_baseline_batch.sh --baseline a
#   bash scripts/cluster/submit_baseline_batch.sh --baseline a --start 0 --end 69
#
# Detached (VPN-safe):
#   bash scripts/cluster/start_baseline_batch_daemon.sh --baseline a

set -euo pipefail

START=0
END=""
END_SET=0
POLL_SEC=30
MAX_OPPORTUNISTIC_JOBS=2
BASELINE="a"
SPLIT="test"
BACKEND="qwen"

PINNED_REPO="${MLMI_PINNED_REPO:-/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen}"
export MLMI_PINNED_REPO="${PINNED_REPO}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2 ;;
    --end) END="$2"; END_SET=1; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    -h | --help)
      echo "Usage: $0 [--baseline a|b1|b2|b2_cap|p0|p1|p2|p3|naive] [--split test] [--backend qwen] [--start N] [--end N]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "${PINNED_REPO}"

# Validate baseline key against the Python registry before queueing jobs.
VALID_BASELINES="$(
  PYTHONPATH="${PINNED_REPO}${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<'PY'
from scripts.inference.run_baseline_batch import BASELINES
print(" ".join(sorted(BASELINES)))
PY
)"
# shellcheck disable=SC2086
if ! printf '%s\n' ${VALID_BASELINES} | grep -qx -- "${BASELINE}"; then
  echo "Unknown BASELINE='${BASELINE}'. Valid: ${VALID_BASELINES}" >&2
  exit 2
fi

if [[ "${END_SET}" -eq 0 ]]; then
  CASE_COUNT="$(
    PYTHONPATH="${PINNED_REPO}${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<PY
from pathlib import Path
from extraction.case_ids import load_cases_from_chains
chains = Path("${PINNED_REPO}") / "data" / "labels" / "chains.jsonl"
cases = load_cases_from_chains(chains, split="${SPLIT}")
print(len(cases))
PY
  )"
  if [[ -z "${CASE_COUNT}" || "${CASE_COUNT}" -lt 1 ]]; then
    echo "No cases found for split=${SPLIT} in ${PINNED_REPO}/data/labels/chains.jsonl" >&2
    exit 1
  fi
  END=$((CASE_COUNT - 1))
  echo "Derived END=${END} from ${CASE_COUNT} ${SPLIT} cases" >&2
fi

if (( END < START )); then
  echo "END (${END}) must be >= START (${START})" >&2
  exit 2
fi

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

JOB_NAME="path-baseline-${BASELINE}-${SPLIT}"
submitted=0
failed=0
echo "Batch submitter: baseline=${BASELINE} split=${SPLIT} backend=${BACKEND} case-index ${START}-${END}" >&2
echo "Pinned repo: ${PINNED_REPO}" >&2
echo "Job name: ${JOB_NAME} (max opportunistic jobs=${MAX_OPPORTUNISTIC_JOBS})" >&2

for idx in $(seq "${START}" "${END}"); do
  wait_for_slot

  if ! out=$(
    sbatch \
      --export=NONE,BASELINE="${BASELINE}",SPLIT="${SPLIT}",BACKEND="${BACKEND}",MLMI_PINNED_REPO="${PINNED_REPO}" \
      --array="${idx}" \
      --job-name="${JOB_NAME}" \
      scripts/cluster/run_baseline_batch.sh
  ); then
    echo "sbatch failed for case-index ${idx}" >&2
    failed=$((failed + 1))
    continue
  fi
  echo "${out} (case-index ${idx})"
  submitted=$((submitted + 1))
done

echo "Done: submitted ${submitted} tasks (case-index ${START}-${END}); failed_submits=${failed}."
if (( failed > 0 && submitted == 0 )); then
  exit 1
fi
