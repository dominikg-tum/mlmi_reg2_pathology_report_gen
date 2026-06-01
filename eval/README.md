# Evaluation

```bash
python -m eval.run_eval --pred runs/b2/predictions.jsonl --gt data/labels/chains.jsonl --split test
```

Prediction JSONL format (one case per line):

```json
{"slide_id": "abc.svs", "chain-of-thought": [{"question": "...", "answer": "...", "next_question": "..."}], "report": "..."}
```

Metrics: Binary Path Validity, Edge-F1, MESS, ROUGE-L, BLEU-4, clinical proxy.
