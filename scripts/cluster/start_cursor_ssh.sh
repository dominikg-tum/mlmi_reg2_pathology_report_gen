#!/bin/bash
#SBATCH --job-name=cursor-ssh
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/cursor_ssh_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/cursor_ssh_%j.err

# CPU-only Remote SSH for Cursor / VS Code on a compute node.
#
# Do NOT add --gres=gpu here. Cursor is an editor session — it does not use the GPU.
# The cluster example /mnt/general/examples/ssh.sh requests a GPU and will trigger
# idle-GPU warnings (auto-cancel after ~2 h at 0% util).
#
# GPU preprocessing / inference: submit separate jobs, e.g.
#   sbatch scripts/cluster/run_offline_wsi.sh
#
# Usage (from head node):
#   sbatch scripts/cluster/start_cursor_ssh.sh
#   sleep 15 && tail -20 /mnt/projects/mlmi/reg2/dominik/logs/cursor_ssh_<JOBID>.out

set -euo pipefail

# shellcheck source=load_paths.sh
source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths

mkdir -p "${LOGS_DIR}"

PORT=$(python3 -c "import random; print(random.randint(20000, 30000))")
HOSTNAME=$(hostname)

echo "JOB ID: ${SLURM_JOB_ID}"
echo "Node: ${HOSTNAME}"
echo "Partition: ${SLURM_JOB_PARTITION:-24g} (CPU only — no GPU allocated)"

start-ssh-server "${PORT}"

echo ""
echo "Connect with:"
echo "  ssh -p ${PORT} ${USER}@${HOSTNAME}"
echo ""
echo "Cursor / VS Code folder URI:"
echo "  code --folder-uri vscode-remote://ssh-remote+${USER}@${HOSTNAME}:${PORT}${REPO}"
echo ""
echo "Open folder: ${REPO}"

sleep 48h
