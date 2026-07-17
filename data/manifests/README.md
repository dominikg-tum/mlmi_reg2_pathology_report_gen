# Case / WSI manifests

## `wsi_name_map.csv` (required for offline jobs)

UUID on-disk name ↔ canonical `TUM_Uterus_XXXX.svs` (matches thumbnail stems).

| Column | Description |
|--------|-------------|
| `wsi_index` | SLURM `--array` index (`0` … `463`) |
| `slide_id` | Canonical cache / thumbnail id (`TUM_Uterus_XXXX.svs`) |
| `disk_name` | UUID `.svs` under `cluster.data_dir` |
| `case_key` | Patient/case key (`p0001`, …) |
| `report_duplicate` | `1` if row was SKIPPED in the name-mapping xlsx (still encode the WSI) |

Rebuild from xlsx:

```bash
uv run python -m scripts.data.build_wsi_name_map
```

## `cases.csv`

| Column | Description |
|--------|-------------|
| `case_id` | Case key from the name map (`p0001`, …) |
| `slide_ids` | Comma-separated canonical `.svs` ids |
| `case_class` | A–E richness (confirm with Han) |
| `disease_label` | From name-mapping sheet |
| `split` | `train` / `test` — test never in RAG index |
| `n_slides` | Slide count |

```bash
uv run python -m scripts.data.build_manifest
```

Example: `cases.csv.example`
