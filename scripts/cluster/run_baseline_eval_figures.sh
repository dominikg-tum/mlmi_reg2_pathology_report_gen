#!/bin/bash
# Regenerate baseline eval figures (no Jupyter required).
#SBATCH --job-name=baseline-eval-figures
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/baseline_eval_figures_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/baseline_eval_figures_%j.err

set -euo pipefail

# shellcheck source=load_paths.sh
source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths

PYDEPS="${WORK_DIR}/tmp/pip_target"
mkdir -p "${PYDEPS}" "${WORK_DIR}/tmp/mplconfig" "${LOGS_DIR}"
export MPLCONFIGDIR="${WORK_DIR}/tmp/mplconfig"
export MPLBACKEND=Agg
export PYTHONPATH="${PYDEPS}:${REPO}:${PYTHONPATH:-}"

pip install -q --target "${PYDEPS}" matplotlib seaborn pandas pyyaml 2>/dev/null || true

python "${REPO}/scripts/analysis/run_baseline_eval_figures.py"
echo "Figures -> ${REPO}/notebooks/figures/"
