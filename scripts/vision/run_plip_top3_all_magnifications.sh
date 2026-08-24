#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/mnt/research/ljs/mlmi_reg2_pathology_report_gen}"
DATA_DIR="${DATA_DIR:-/mnt/research/data_slow/miccai_challenge/slides_with_public_id_final}"
PLIP_MODEL="${PLIP_MODEL:-$PROJECT/plip}"
OUT_ROOT="${OUT_ROOT:-$PROJECT/runs/plip_top3_all_magnifications}"
CONDA="${CONDA:-/home/ge54xof/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-qwen25}"
GPU="${GPU:-2}"

# Full extraction default: 0 means all tissue patches.
# For quick tests, override e.g. MAX_PATCHES=32 LIMIT=1.
MAX_PATCHES="${MAX_PATCHES:-0}"
LIMIT="${LIMIT:-0}"
TOP_K="${TOP_K:-3}"
PATCH_SIZE="${PATCH_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NODE_LIMIT="${NODE_LIMIT:-0}"
ASK_VLM_LIMIT="${ASK_VLM_LIMIT:-0}"
BACKGROUND_THRESHOLD="${BACKGROUND_THRESHOLD:-220}"
LEVELS="${LEVELS:-1x 1.25x 2.5x 5x 10x}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SAVE_ALL_PATCHES="${SAVE_ALL_PATCHES:-0}"

mkdir -p "$OUT_ROOT/logs"
cd "$PROJECT"

echo "Project: $PROJECT"
echo "Dataset (read-only): $DATA_DIR"
echo "PLIP model: $PLIP_MODEL"
echo "Output root: $OUT_ROOT"
echo "Levels: $LEVELS"
echo "MAX_PATCHES=$MAX_PATCHES LIMIT=$LIMIT TOP_K=$TOP_K"
echo "SKIP_EXISTING=$SKIP_EXISTING SAVE_ALL_PATCHES=$SAVE_ALL_PATCHES"

extra_args=()
if [ "$SKIP_EXISTING" = "1" ]; then
  extra_args+=(--skip-existing)
fi
if [ "$SAVE_ALL_PATCHES" = "1" ]; then
  extra_args+=(--save-all-patches)
fi

for level in $LEVELS; do
  safe_level="${level//./p}"
  safe_level="${safe_level//×/x}"
  log="$OUT_ROOT/logs/plip_top3_${safe_level}_$(date +%Y%m%d_%H%M%S).log"
  echo "== Running level $level =="
  echo "Log: $log"
  PYTHONPATH="$PROJECT" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$GPU" \
  "$CONDA" run -n "$CONDA_ENV" python scripts/vision/plip_topk_retrieve_and_vlm.py \
    --data-dir "$DATA_DIR" \
    --plip-model "$PLIP_MODEL" \
    --out-root "$OUT_ROOT" \
    --limit "$LIMIT" \
    --level "$level" \
    --patch-size "$PATCH_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --max-patches "$MAX_PATCHES" \
    --top-k "$TOP_K" \
    --background-threshold "$BACKGROUND_THRESHOLD" \
    --node-limit "$NODE_LIMIT" \
    --ask-vlm-limit "$ASK_VLM_LIMIT" \
    "${extra_args[@]}" \
    2>&1 | tee "$log"
done

echo "Done. Outputs written to $OUT_ROOT"
