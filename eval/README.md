# Evaluation

```bash
python -m eval.run_eval --pred runs/b2/predictions.jsonl --gt data/labels/chains.jsonl --split test
```

`--split` filters on the `split` field in the **ground-truth** JSONL (e.g. only `test` cases). Predictions are matched by `slide_id`.

Prediction JSONL format (one case per line):

```json
{"slide_id": "abc.svs", "chain-of-thought": [{"question": "...", "answer": "...", "next_question": "..."}], "report": "..."}
```

Metrics: Binary Path Validity, Edge-F1, MESS, ROUGE-L, BLEU-4, clinical proxy.
