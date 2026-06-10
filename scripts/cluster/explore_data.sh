#!/bin/bash
#SBATCH --job-name=mlmi-explore
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/explore_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/explore_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@tum.de

set -euo pipefail

# shellcheck source=load_paths.sh
source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths

mkdir -p "${LOGS_DIR}"

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${CONTAINER}" \
  jupyter nbconvert --to notebook --execute --inplace \
    "${REPO}/notebooks/explore_wsi.ipynb"
