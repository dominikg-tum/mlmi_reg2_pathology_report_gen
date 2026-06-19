#!/bin/bash
#SBATCH --job-name=wp3-smoke-test
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:0
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/wp3_smoke_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/wp3_smoke_%j.err
#
# Prerequisite: vLLM running on same node (localhost:8000).
# Submit with nodelist matching the qwen-server job:
#   sbatch --nodelist=heidelberg scripts/cluster/smoke_test_chains.sh

set -euo pipefail

source scripts/cluster/load_paths.sh
load_cluster_paths

curl -sf http://localhost:8000/v1/models | python3 -m json.tool | head -20
echo "--- smoke test ---"

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
    pip install -q openai pandas pyyaml tqdm openpyxl 2>/dev/null || true
    python -m extraction.qa_extractor
  "

echo "SMOKE_OK"
