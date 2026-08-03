# Manifest — LoRA v1 artifacts

Each row: **local path** (under repo) ← **cluster source** (if applicable).

## Training curve

| Local | Cluster source | Notes |
|-------|----------------|-------|
| `training/train_loss.csv` | derived from `trainer_state.json` | 32 log points |
| `training/train_loss.png` | `/mnt/home/dogakonuk/lora/eval/plots/train_loss.png` (optional) | also regenerated from CSV |
| `training/train_loss_summary.json` | n/a (generated) | first/last/min loss, epochs |
| *(reference only)* | `/mnt/home/dogakonuk/lora/qwen3vl-uterus/adapter/checkpoint-160/trainer_state.json` | canonical HF log |

Also kept under `training/artifacts/lora_v1/` for backward compatibility (same files).

## Node-level eval (prebuilt multimodal samples)

| Local | Cluster source |
|-------|----------------|
| `node_eval/base_report.json` | `/mnt/home/dogakonuk/lora/eval/base_report.json` |
| `node_eval/finetuned_report.json` | `/mnt/home/dogakonuk/lora/eval/finetuned_report.json` |

Headline: overall answer-key accuracy **0.596 → 0.691** (+9.6 pts).

## Full-agent ablation (68 test cases)

Cluster runs root: `/mnt/projects/mlmi/reg2/dogakonuk/runs/`

| Local | Cluster source |
|-------|----------------|
| `ablation/arms/baseline_a_flat/metrics.json` | `.../baseline_a_flat/metrics.json` |
| `ablation/arms/baseline_a_flat_lora/metrics.json` | `.../baseline_a_flat_lora/metrics.json` |
| `ablation/arms/baseline_p0_patch_cosine/metrics.json` | `.../baseline_p0_patch_cosine/metrics.json` |
| `ablation/arms/baseline_p0_patch_cosine_lora/metrics.json` | `.../baseline_p0_patch_cosine_lora/metrics.json` |
| `ablation/arms/<arm>/predictions.jsonl` | same arm dir (optional; often gitignored) |
| `ablation/plots/ablation_cot_metrics.png` | `.../runs/plots/ablation_cot_metrics.png` |
| `ablation/plots/ablation_report_metrics.png` | `.../runs/plots/ablation_report_metrics.png` |
| `ablation/tables/ablation_metrics_summary.csv` | `.../runs/plots/ablation_metrics_summary.csv` |
| `ablation/tables/ablation_metrics_summary.json` | `.../runs/plots/ablation_metrics_summary.json` |
| `ablation/tables/ablation_rescore_table.json` | `.../runs/plots/ablation_rescore_table.json` |

Scoring notes (rescored Phase 1):

- CoT path metrics **exclude** the trailing `report` node (BPV ends at `diagnosis`).
- **Final Diag Acc** = exact match of the graph `diagnosis` node `answer_key` vs GT.
- Report metrics: ROUGE-L / BLEU-4 / Clinical (proxy) / Num. FID / Negation Cons.
  (BERTScore skipped in the light rescore unless re-run without `--skip-bert`).
- Re-run `scripts/training/rescore_ablation_arms.py` after pulling so `metrics.json` /
  CoT plot Final Diag Acc bars use the exact node-key definition.

## Adapter weights (never in this tree)

| Purpose | Path |
|---------|------|
| Shared for teammates | `/mnt/projects/mlmi/reg2/dogakonuk/lora/qwen3vl-uterus/adapter/` |
| Private + checkpoints | `/mnt/home/dogakonuk/lora/qwen3vl-uterus/adapter/` |

## Optional logs

| Local | Cluster source |
|-------|----------------|
| `logs/` (gitignored) | `/mnt/home/dogakonuk/logs/lora_train_*.out`, `lora_eval_*.out` |
| | `/mnt/projects/mlmi/reg2/dogakonuk/logs/` (baseline batch) |

## Populate status

After `bash scripts/local/fetch_lora_v1_artifacts.sh`, every row marked “Cluster source” above
should exist locally except optional predictions/logs.
