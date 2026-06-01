# Training (DOMI)

1. WP3 chains → `data/labels/chains.jsonl`
2. `training/dataset.py` — build samples with **same** `--visual` / `--retriever` as inference
3. `training/lora.py` — LoRA run on cluster
4. Register `FineTunedBackend` in `agent/backends.py`

See Patho-R1 for pathology CoT SFT practices.
