#!/usr/bin/env bash
set -euo pipefail

cd /mnt/research/ljs/mlmi_reg2_pathology_report_gen

echo -n "pid="
cat runs/plip_top3_all_magnifications/full_run.pid
echo

echo "processes"
ps -u ge54xof -o pid,ppid,stat,etime,cmd | grep -E "plip_topk|run_plip_top3|conda run" | grep -v grep || true

echo "counts"
echo -n "dataset_svs="
find /mnt/research/data_slow/miccai_challenge/slides_with_public_id_final -maxdepth 1 -type f -name "*.svs" | wc -l
echo -n "summaries="
find runs/plip_top3_all_magnifications -name plip_topk_summary.json | wc -l
echo -n "no_tissue="
python - <<'PY'
import json
from pathlib import Path
root = Path("runs/plip_top3_all_magnifications")
count = 0
for path in root.rglob("plip_topk_summary.json"):
    try:
        if json.loads(path.read_text()).get("status") == "no_tissue_patches":
            count += 1
    except Exception:
        pass
print(count)
PY
echo -n "topk_jpgs="
find runs/plip_top3_all_magnifications -path "*/topk/*/*.jpg" | wc -l
echo -n "patch_dirs="
find runs/plip_top3_all_magnifications -type d -name patches | wc -l

echo "latest_log"
ls -t runs/plip_top3_all_magnifications/logs/plip_top3_*.log | head -1

echo "log_tail"
tail -50 "$(ls -t runs/plip_top3_all_magnifications/logs/plip_top3_*.log | head -1)"
