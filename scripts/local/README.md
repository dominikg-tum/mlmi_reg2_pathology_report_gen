# Submit SLURM jobs from your laptop or cluster head

Pinned code path (NFS quota-safe):

`/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen`

Cache (CONCH 20x): `/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/cache_20x_v2`

**Done artifact per slide:** `patch_embeddings_20x.pt` (TITAN `slide_embedding.pt` is **off** by default).

---

## Run full batch without VPN (recommended)

The batch submitter must run **on the cluster head**, not your laptop.

### Step 1 — Laptop terminal (VPN on, one time)

Sync latest code:

```bash
cd "/home/garstenauer/Documents/orga/other/MLMI REG2 PATH/mlmi_reg2_pathology_report_gen"
bash scripts/local/sync_repo_to_cluster.sh
```

Stop any **laptop** batch submitter if still running (Ctrl+C in that terminal).

SSH to head:

```bash
ssh dominikgarstenauer@head.garching.camp.cluster
```

### Step 2 — Head terminal (same SSH session)

Kill stale locks / old laptop submitter if needed:

```bash
rm -f /mnt/projects/mlmi/TUMUntera/dominik_garstenauer/locks/wsi_batch_local.lock
pkill -f 'submit_offline_wsi_batch_remote' 2>/dev/null || true
```

Find first incomplete index (optional):

```bash
find /mnt/projects/mlmi/TUMUntera/dominik_garstenauer/cache_20x_v2 -maxdepth 2 -name patch_embeddings_20x.pt | wc -l
```

Start detached batch submitter (replace `--start` with your count):

```bash
cd /mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen
bash scripts/cluster/start_offline_wsi_batch_daemon.sh --start 0
```

You should see `Started batch submitter pid=...` and a log path.

### Step 3 — Disconnect VPN

Safe now. SLURM GPU jobs and the head-node submitter keep running.

### Step 4 — Monitor later (VPN on again, any terminal)

```bash
ssh dominikgarstenauer@head.garching.camp.cluster
tail -f /mnt/projects/mlmi/TUMUntera/dominik_garstenauer/logs/wsi_batch_submitter.log
squeue -u dominikgarstenauer -n wsi-offline-pipeline
find /mnt/projects/mlmi/TUMUntera/dominik_garstenauer/cache_20x_v2 -name patch_embeddings_20x.pt | wc -l
```

---

## One slide smoke test (laptop + VPN)

```bash
bash scripts/local/submit_offline_wsi_remote.sh --wsi-index 0
```

---

## Laptop batch (needs VPN the whole time)

Only use if head daemon is **not** running:

```bash
bash scripts/local/submit_offline_wsi_batch_remote.sh --handoff --start 0
```

---

## Optional: TITAN slide embedding (Phase 2)

Not run in the default pipeline. To add later for one slide:

```bash
python -m scripts.vision.encode_slide_embeddings --wsi-index N
```

Or full pipeline with slide emb:

```bash
python -m scripts.preprocess.run_offline_wsi --wsi-index N --with-slide-emb
```
