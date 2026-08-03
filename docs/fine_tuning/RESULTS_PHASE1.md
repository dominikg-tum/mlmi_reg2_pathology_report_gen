# Phase 1 results — LoRA v1

Companion to [`FINE_TUNING_PROCESS.md`](FINE_TUNING_PROCESS.md).  
Machine-readable copies: [`artifacts/lora_v1/`](../../artifacts/lora_v1/).

## Training

| Item | Value |
|------|------:|
| Epochs / steps / log points | 2 / 160 / 32 |
| First → last train loss | 1.488 → 0.045 |
| Min train loss | 0.017 |
| Val / eval_loss | none logged |

Files: `artifacts/lora_v1/training/train_loss.{csv,png}` + `train_loss_summary.json`.

## Node-level answer_key accuracy (prebuilt test samples)

| Model | Overall accuracy |
|-------|-----------------:|
| Base Qwen3-VL-8B | 0.596 |
| LoRA v1 | **0.691** (+9.6 pts) |

Source JSON: `artifacts/lora_v1/node_eval/{base,finetuned}_report.json`  
(largest gains historically on `compartment`, `stage_extent`, `microscopic_pattern`, etc. —
see per-node breakdown in those reports).

## Full-agent ablation (n = 68 test cases)

Rescored with **report node excluded** from CoT path metrics.
**Final Diag Acc** = exact match on the graph `diagnosis` node answer vs GT.
Re-run `scripts/training/rescore_ablation_arms.py` after pulling to refresh that column / CoT plot.

| Arm | BPV | Edge-F1 | Node Acc | Diag Acc | ROUGE-L | Num FID | Negation |
|-----|----:|--------:|---------:|---------:|--------:|--------:|---------:|
| a base | 0.044 | 0.236 | 0.247 | 0.412 | 0.045 | 0.000 | 0.675 |
| a LoRA | 0.294 | 0.508 | 0.502 | 0.544 | 0.024 | 0.000 | 0.909 |
| p0 base | 0.074 | 0.226 | 0.274 | 0.088 | 0.039 | 0.042 | 0.235 |
| p0 LoRA | **0.338** | **0.531** | **0.511** | **0.559** | 0.016 | 0.000 | 1.000 |

Report plot also includes BLEU-4 and Clin F1 (often ≈ ROUGE-L here). BERTScore was skipped in the light rescore.

**Takeaways for slides**

- LoRA improves CoT path metrics on both **a** and **p0** (~+0.25 Edge-F1 / node acc).
- Best CoT arm: **p0 + LoRA**.
- Report ROUGE stays low; LoRA does not help report text (no report SFT).
- BPV was previously stuck at 0 when GT omitted `report` but preds included it; rescored BPV is meaningful.

Plots (separate, as requested by the team):

- `artifacts/lora_v1/ablation/plots/ablation_cot_metrics.png`
- `artifacts/lora_v1/ablation/plots/ablation_report_metrics.png`

Tables:

- `artifacts/lora_v1/ablation/tables/ablation_metrics_summary.csv`
- `artifacts/lora_v1/ablation/tables/ablation_rescore_table.json`
