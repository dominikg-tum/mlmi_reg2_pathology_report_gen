#!/bin/bash
#SBATCH --job-name=cursor-ssh
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=12g
#SBATCH --qos=students_normal
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/cursor_ssh_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/cursor_ssh_%j.err

# CPU-only Remote SSH for Cursor / VS Code on a compute node.
#
# Cluster policy (SLURM GPU Cluster Garching Users wiki):
#   - HEAD: submit jobs only — no IDEs, no heavy compute.
#   - Remote SSH / VS Code: allowed via sbatch on a compute node (see /mnt/general/examples/ssh.sh).
#   - 24g: jobs that do not use a GPU for 2 h are auto-cancelled — do NOT run a GPU-less editor there.
#   - Split work: editor session (this job, CPU only) + separate GPU sbatch for training/preprocessing.
#
# Do NOT use /mnt/general/examples/ssh.sh for Cursor — it requests --gres=gpu:1 and wastes a GPU
# at 0% util (24g idle-GPU auto-cancel after ~2 h).
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
HOSTNAME=$(hostname -s)
FQDN="${HOSTNAME}.garching.camp.cluster"

echo "JOB ID: ${SLURM_JOB_ID}"
echo "Node: ${FQDN}"
echo "Partition: ${SLURM_JOB_PARTITION:-12g} (CPU only — no GPU allocated)"
echo "QOS: ${SLURM_JOB_QOS:-students_normal}"

start-ssh-server "${PORT}"

echo ""
echo "Connect with:"
echo "  ssh -p ${PORT} ${USER}@${FQDN}"
echo ""
echo "Cursor / VS Code folder URI:"
echo "  cursor --folder-uri vscode-remote://ssh-remote+${USER}@${FQDN}:${PORT}${REPO}"
echo ""
echo "Remote-SSH host string (paste in Connect to Host):"
echo "  ${USER}@${FQDN}:${PORT}"
echo ""
echo "Open folder: ${REPO}"

sleep 48h
