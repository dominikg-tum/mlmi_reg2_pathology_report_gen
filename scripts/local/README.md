# Submit SLURM jobs from your laptop

Run offline WSI jobs using **your feature-branch code** without relying on the team shared repo checkout at `/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen`.

## How it works

1. **Rsync** your local repo → `/mnt/projects/mlmi/reg2/dominik/repos/mlmi_reg2_pathology_report_gen` (pinned path on NFS).
2. **SSH** to `head` and `sbatch` `scripts/cluster/run_offline_wsi_pinned.sh`.
3. Jobs still use the same **cache**, **logs**, and **data** paths (`configs/paths.yaml`).

Teammates can `git checkout main` on the shared repo — your pinned jobs are unaffected.

## One-time laptop setup

1. TUM VPN + SSH key to head (see `docs/cluster_setup.md` §6).
2. Optional overrides in `scripts/local/cluster_env.local.sh`:

```bash
CLUSTER_SSH_HOST="dominikgarstenauer@head.garching.camp.cluster"
PINNED_REPO="/mnt/projects/mlmi/reg2/dominik/repos/mlmi_reg2_pathology_report_gen"
```

## Commands

```bash
# Sync code only
bash scripts/local/sync_repo_to_cluster.sh

# One slide
bash scripts/local/submit_offline_wsi_remote.sh --wsi-index 117

# Batch loop (after handoff — see below)
bash scripts/local/submit_offline_wsi_batch_remote.sh --start 117 --end 459 --handoff

# Status from laptop
ssh dominikgarstenauer@head 'squeue -u dominikgarstenauer'
```

## Automatic handoff (recommended)

When the cluster batch is still running but you want pinned code **without** duplicate slides:

```bash
# Foreground (from laptop, VPN on)
bash scripts/local/auto_handoff_wsi_batch.sh

# Or detach
nohup bash scripts/local/auto_handoff_wsi_batch.sh >> ~/wsi_handoff.log 2>&1 &
tail -f ~/wsi_handoff.log
```

What it does:

1. Reads **current** `wsi-offline-pipeline` job(s) from `squeue` (array task = wsi-index).
2. **Immediately kills** `submit_offline_wsi_batch.sh` on head (so slide N+1 is never submitted on shared repo).
3. **Waits** for the in-flight job to finish (does **not** `scancel` it).
4. Rsyncs your laptop repo → pinned path.
5. Starts local batch from the **first incomplete** wsi-index (cache scan on NFS).

This works on the Garching SLURM cluster — no special scheduler feature; only SSH + `squeue` polling.

## Coexistence with cluster `submit_offline_wsi_batch.sh`

| Situation | What to do |
|-----------|------------|
| **Cluster batch still running** (your current case) | Run **`auto_handoff_wsi_batch.sh`** from laptop when ready — or let cluster batch finish. |
| **Worth switching mid-run?** | **Yes, with auto-handoff** — one in-flight job completes (~20 min), then pinned path takes over. |
| **Ready to hand off manually** | Wait for current SLURM job to finish, then: `ssh head "pkill -f 'bash scripts/cluster/submit_offline_wsi_batch.sh'"`, then local batch with `--handoff --start N`. |
| **Teammate breaks shared repo** | Running jobs keep going; **cluster batch's next slides** may break. After handoff, pinned path protects you. |

**Adding these scripts does not stop or change the running cluster batch submitter** unless you start the local batch (with `--handoff`) or kill the cluster process yourself.

## Monitoring without Cursor

```bash
ssh dominikgarstenauer@head 'squeue -u dominikgarstenauer'
ssh dominikgarstenauer@head 'tail -20 /mnt/projects/mlmi/reg2/dominik/logs/offline_*_116.out'
```
