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

# Keep this list in sync with scripts/inference/run_baseline_batch.py BASELINES.
# Do NOT import that module on the head node: it pulls torch/VLM deps and can hang.
VALID_BASELINES="a b1 b2 b2_cap naive p0 p1 p2 p3"
# shellcheck disable=SC2086
if ! printf '%s\n' ${VALID_BASELINES} | grep -qx -- "${BASELINE}"; then
  echo "Unknown BASELINE='${BASELINE}'. Valid: ${VALID_BASELINES}" >&2
  exit 2
fi

if [[ "${END_SET}" -eq 0 ]]; then
  # Prefer a cheap line count over importing extraction on a loaded head node.
  # Fallback to python only if the jsonl is missing the expected shape.
  CHAINS_FILE="${PINNED_REPO}/data/labels/chains.jsonl"
  if [[ ! -f "${CHAINS_FILE}" ]]; then
    echo "Missing ${CHAINS_FILE}" >&2
    exit 1
  fi
  CASE_COUNT="$(
    PYTHONPATH="${PINNED_REPO}${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<PY
from pathlib import Path
from extraction.case_ids import load_cases_from_chains
cases = load_cases_from_chains(Path("${CHAINS_FILE}"), split="${SPLIT}")
print(len(cases))
PY
  )"
  if [[ -z "${CASE_COUNT}" || "${CASE_COUNT}" -lt 1 ]]; then
    echo "No cases found for split=${SPLIT} in ${CHAINS_FILE}" >&2
    exit 1
  fi
  END=$((CASE_COUNT - 1))
  echo "Derived END=${END} from ${CASE_COUNT} ${SPLIT} cases" >&2
fi

if (( END < START )); then
  echo "END (${END}) must be >= START (${START})" >&2
  exit 2
fi

# SLURM parks jobs as "user env retrieval failed requeued held" when --export=NONE
# env lookup fails on a loaded head. Held jobs never run but still count against the
# QOS cap, which would stall this loop forever.
release_held_jobs() {
  local held job
  held=$(squeue -u "${USER}" -h -o '%i %r' 2>/dev/null \
    | awk '/held/ {print $1}')
  for job in ${held}; do
    if scontrol release "${job}" 2>/dev/null; then
      echo "Released held job ${job}" >&2
    fi
  done
}

wait_for_slot() {
  local waited=0
  while true; do
    local opp_jobs
    # Only RUNNING/PENDING hold the QOS cap; COMPLETING (CG) must not block forever.
    opp_jobs=$(squeue -u "${USER}" -h --qos=students_opportunistic -t RUNNING,PENDING 2>/dev/null | wc -l)
    if ((opp_jobs < MAX_OPPORTUNISTIC_JOBS)); then
      return 0
    fi
    release_held_jobs
    echo "Waiting for students_opportunistic slot (${opp_jobs}/${MAX_OPPORTUNISTIC_JOBS})..."
    sleep "${POLL_SEC}"
    waited=$((waited + 1))
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
      --export=ALL,BASELINE="${BASELINE}",SPLIT="${SPLIT}",BACKEND="${BACKEND}",MLMI_PINNED_REPO="${PINNED_REPO}" \
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
