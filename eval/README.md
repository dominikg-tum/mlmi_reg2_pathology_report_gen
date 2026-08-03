# Evaluation

```bash
python -m eval.run_eval --pred runs/b2/predictions.jsonl --gt data/labels/chains.jsonl --split test
```

`--split` filters on the `split` field in the **ground-truth** JSONL (e.g. only `test` cases). Predictions are matched by `slide_id`.

Prediction JSONL format (one case per line):

```json
{"slide_id": "abc.svs", "chain-of-thought": [{"question": "...", "answer": "...", "next_question": "..."}], "report": "..."}
```

Metrics:

- **CoT** (by default the trailing `report` node is excluded from path metrics): Binary Path
  Validity, Edge-F1, MESS, node accuracy, final diagnosis accuracy (diagnosis-node exact match)
- **Report:** ROUGE-L, BLEU-4, clinical proxy, optional BERTScore, numeric fidelity, negation

```bash
python -m eval.run_eval --pred ... --gt ... --split test --skip-bert --json-out metrics.json
```

Phase-1 LoRA ablation artifacts and plots: [`artifacts/lora_v1/`](../artifacts/lora_v1/).
