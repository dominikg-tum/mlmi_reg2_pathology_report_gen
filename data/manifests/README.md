# Case manifest

| Column | Description |
|--------|-------------|
| `case_id` | Unique case identifier |
| `slide_ids` | Comma-separated `.svs` ids |
| `case_class` | A–E richness (confirm with Han) |
| `split` | `train` / `test` — test never in RAG index; add `val` later if LoRA needs a hold-out |
| `n_slides` | Slide count |

Generate: `python scripts/data/build_manifest.py`

Example: `cases.csv.example`
