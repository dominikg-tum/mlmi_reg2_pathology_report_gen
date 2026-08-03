#!/usr/bin/env bash
# Pull LoRA v1 presentation/report artifacts from the cluster into artifacts/lora_v1/.
#
# Usage (laptop, VPN + SSH):
#   CLUSTER_SSH_HOST=dogakonuk@head.garching.camp.cluster bash scripts/local/fetch_lora_v1_artifacts.sh
#   bash scripts/local/fetch_lora_v1_artifacts.sh --with-predictions
#   bash scripts/local/fetch_lora_v1_artifacts.sh --with-logs
#
# Does NOT download adapter weights.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${SCRIPT_DIR}/cluster_env.sh"

WITH_PREDS=0
WITH_LOGS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-predictions) WITH_PREDS=1; shift ;;
    --with-logs) WITH_LOGS=1; shift ;;
    -h | --help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST="${ROOT}/artifacts/lora_v1"
RUNS_ROOT="/mnt/projects/mlmi/reg2/dogakonuk/runs"
HOME_LORA="/mnt/home/dogakonuk/lora"
HOME_LOGS="/mnt/home/dogakonuk/logs"
PROJ_LOGS="/mnt/projects/mlmi/reg2/dogakonuk/logs"

ARMS=(
  baseline_a_flat
  baseline_a_flat_lora
  baseline_p0_patch_cosine
  baseline_p0_patch_cosine_lora
)

mkdir -p \
  "${DEST}/training" \
  "${DEST}/node_eval" \
  "${DEST}/ablation/plots" \
  "${DEST}/ablation/tables" \
  "${DEST}/logs"

echo "SSH host: ${CLUSTER_SSH_HOST}"
echo "Destination: ${DEST}"

_scp() {
  scp -o ForwardX11=no "$@"
}

echo "==> node_eval reports"
_scp \
  "${CLUSTER_SSH_HOST}:${HOME_LORA}/eval/base_report.json" \
  "${CLUSTER_SSH_HOST}:${HOME_LORA}/eval/finetuned_report.json" \
  "${DEST}/node_eval/"

echo "==> ablation metrics.json"
for arm in "${ARMS[@]}"; do
  mkdir -p "${DEST}/ablation/arms/${arm}"
  _scp \
    "${CLUSTER_SSH_HOST}:${RUNS_ROOT}/${arm}/metrics.json" \
    "${DEST}/ablation/arms/${arm}/"
  if [[ "${WITH_PREDS}" -eq 1 ]]; then
    _scp \
      "${CLUSTER_SSH_HOST}:${RUNS_ROOT}/${arm}/predictions.jsonl" \
      "${DEST}/ablation/arms/${arm}/" || true
  fi
done

echo "==> ablation plots + tables"
_scp \
  "${CLUSTER_SSH_HOST}:${RUNS_ROOT}/plots/ablation_cot_metrics.png" \
  "${CLUSTER_SSH_HOST}:${RUNS_ROOT}/plots/ablation_report_metrics.png" \
  "${DEST}/ablation/plots/" || {
    echo "WARN: plot PNGs missing on cluster — regenerate with plot_ablation_metrics.py" >&2
  }

_scp \
  "${CLUSTER_SSH_HOST}:${RUNS_ROOT}/plots/ablation_metrics_summary.csv" \
  "${CLUSTER_SSH_HOST}:${RUNS_ROOT}/plots/ablation_metrics_summary.json" \
  "${CLUSTER_SSH_HOST}:${RUNS_ROOT}/plots/ablation_rescore_table.json" \
  "${DEST}/ablation/tables/" || true

echo "==> optional cluster train_loss.png (local CSV already in repo)"
_scp \
  "${CLUSTER_SSH_HOST}:${HOME_LORA}/eval/plots/train_loss.png" \
  "${DEST}/training/train_loss_cluster.png" 2>/dev/null || true

if [[ "${WITH_LOGS}" -eq 1 ]]; then
  echo "==> selected logs"
  mkdir -p "${DEST}/logs"
  _scp \
    "${CLUSTER_SSH_HOST}:${HOME_LOGS}/lora_train_15570.out" \
    "${CLUSTER_SSH_HOST}:${HOME_LOGS}/lora_eval_15595.out" \
    "${DEST}/logs/" 2>/dev/null || true
  # Best-effort copy of projects logs dir listing only if small; skip bulk.
  echo "Projects logs remain on cluster: ${PROJ_LOGS}"
fi

# Keep legacy path in sync for train_loss if present
if [[ -f "${DEST}/training/train_loss.csv" ]]; then
  mkdir -p "${ROOT}/training/artifacts/lora_v1"
  cp -f "${DEST}/training/train_loss.csv" "${ROOT}/training/artifacts/lora_v1/" 2>/dev/null || true
fi

echo
echo "Done. Tree:"
find "${DEST}" -type f | sort
echo
echo "Adapter weights were NOT downloaded (by design)."
