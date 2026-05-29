#!/bin/bash
#SBATCH --job-name=qwen-server
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/qwen_server_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/qwen_server_%j.err

set -euo pipefail

mkdir -p /mnt/projects/mlmi/reg2/dominik/logs

CONTAINER=/mnt/projects/mlmi/reg2/containers/qwen25_dev_updated.sqsh
MODEL=/mnt/projects/mlmi/reg2/models/Qwen2.5-7B-Instruct

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${CONTAINER}" \
  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port 8000
