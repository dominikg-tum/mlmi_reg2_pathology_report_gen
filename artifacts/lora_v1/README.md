# LoRA v1 artifacts (Phase 1) — Doga

Canonical, merge-friendly location for **all presentation / report evidence** from the
Qwen3-VL-8B uterus LoRA Phase-1 work.

| Role | Path |
|------|------|
| **This tree** | tracked metrics, plots, tables, small JSON reports |
| **Process write-up** | [`docs/fine_tuning/FINE_TUNING_PROCESS.md`](../../docs/fine_tuning/FINE_TUNING_PROCESS.md) |
| **Results summary** | [`docs/fine_tuning/RESULTS_PHASE1.md`](../../docs/fine_tuning/RESULTS_PHASE1.md) |
| **How to train / serve (code)** | [`training/README.md`](../../training/README.md) |
| **Adapter weights (not in git)** | cluster only — see below |

## Layout

```text
artifacts/lora_v1/
  README.md                 ← you are here
  MANIFEST.md               ← file-by-file inventory + cluster sources
  training/                 ← train-loss curve + CSV + summary
  node_eval/                ← prebuilt-sample answer_key reports (base vs LoRA)
  ablation/
    arms/<run_name>/        ← per-arm metrics.json (+ optional predictions.jsonl)
    plots/                  ← CoT vs Report bar charts (separate PNGs)
    tables/                 ← CSV / JSON summary tables
  logs/                     ← optional SLURM excerpts (gitignored by default)
```

## Populate from the cluster

From the **laptop** (VPN + SSH), with host override if needed:

```powershell
# PowerShell
$env:CLUSTER_SSH_HOST = "dogakonuk@head.garching.camp.cluster"
bash scripts/local/fetch_lora_v1_artifacts.sh
```

Or:

```bash
CLUSTER_SSH_HOST=dogakonuk@head.garching.camp.cluster \
  bash scripts/local/fetch_lora_v1_artifacts.sh
```

What gets pulled is listed in `MANIFEST.md`. Adapter **weights are never fetched into git**.

## What belongs in git vs cluster-only

| In git (`artifacts/lora_v1/`) | Cluster-only (do not commit) |
|-------------------------------|------------------------------|
| Train loss CSV / PNG / summary | LoRA adapter `.safetensors`, checkpoints, optimizer |
| `node_eval/*_report.json` | Full `train_images/`, `test_images/` |
| Ablation `metrics.json` | Merged full-model weights (if any) |
| Ablation plots + summary CSV/JSON | Raw SLURM `.out` (optional; large) |
| Optional: `predictions.jsonl` if small | |

Shared adapter for teammates (inference files):

`/mnt/projects/mlmi/reg2/dogakonuk/lora/qwen3vl-uterus/adapter/`

Private full copy (home):

`/mnt/home/dogakonuk/lora/qwen3vl-uterus/adapter/`

## Arm folder names

| Label | Run directory name |
|-------|--------------------|
| a base | `baseline_a_flat` |
| a LoRA | `baseline_a_flat_lora` |
| p0 base | `baseline_p0_patch_cosine` |
| p0 LoRA | `baseline_p0_patch_cosine_lora` |

Cluster runs root: `/mnt/projects/mlmi/reg2/dogakonuk/runs/`
