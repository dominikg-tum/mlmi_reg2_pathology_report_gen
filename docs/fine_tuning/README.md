# Fine-tuning docs (Doga / LoRA v1)

| Doc | Contents |
|-----|----------|
| [FINE_TUNING_PROCESS.md](FINE_TUNING_PROCESS.md) | End-to-end process for the course report |
| [RESULTS_PHASE1.md](RESULTS_PHASE1.md) | Tables + takeaways for slides |
| [fine-tuning-pres.md](fine-tuning-pres.md) | 4-slide deck outline + speaker notes |
| [../../artifacts/lora_v1/](../../artifacts/lora_v1/) | Metrics, plots, CSVs, node-eval JSON |
| [../../training/README.md](../../training/README.md) | How to rebuild / train / serve (code) |

Populate cluster-produced files:

```bash
CLUSTER_SSH_HOST=dogakonuk@head.garching.camp.cluster \
  bash scripts/local/fetch_lora_v1_artifacts.sh
```
