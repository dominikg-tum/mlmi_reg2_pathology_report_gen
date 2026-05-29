#!/bin/bash
#SBATCH --job-name=mlmi-explore
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/explore_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/explore_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@tum.de

set -euo pipefail

mkdir -p /mnt/projects/mlmi/reg2/dominik/logs

REPO=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
# Update CONTAINER after exporting dominik_mlmi (yourname_YYYYMMDD_base.sqsh)
CONTAINER=/mnt/projects/mlmi/reg2/containers/qwen25_dev_updated.sqsh

# Execute the WP1 exploration notebook headlessly (writes outputs back in-place).
enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${CONTAINER}" \
  jupyter nbconvert --to notebook --execute --inplace \
    "${REPO}/notebooks/explore_wsi.ipynb"
