#!/usr/bin/env bash
set -euo pipefail

cd /mnt/research/ljs/mlmi_reg2_pathology_report_gen
mkdir -p runs/plip_top3_all_magnifications/logs

log="runs/plip_top3_all_magnifications/full_run_$(date +%Y%m%d_%H%M%S).log"

nohup env \
  GPU=2 \
  BATCH_SIZE=512 \
  MAX_PATCHES=0 \
  LIMIT=0 \
  NODE_LIMIT=0 \
  ASK_VLM_LIMIT=0 \
  SKIP_EXISTING=1 \
  SAVE_ALL_PATCHES=0 \
  OUT_ROOT=/mnt/research/ljs/mlmi_reg2_pathology_report_gen/runs/plip_top3_all_magnifications \
  LEVELS="1x 1.25x 2.5x 5x 10x" \
  scripts/vision/run_plip_top3_all_magnifications.sh > "$log" 2>&1 &

pid=$!
echo "$pid" > runs/plip_top3_all_magnifications/full_run.pid
echo "FULL_RUN_PID=$pid"
echo "FULL_RUN_LOG=/mnt/research/ljs/mlmi_reg2_pathology_report_gen/$log"
