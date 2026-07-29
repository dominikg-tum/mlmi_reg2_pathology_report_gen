# Case / WSI manifests

## Authority

| File | Role |
|------|------|
| `wsi_name_map.csv` | Physical WSI inventory (UUID ↔ `TUM_Uterus_XXXX.svs` ↔ `case_key`) |
| `cases.csv` | **Canonical train/test split** and case-level slide lists |

Do **not** re-run `build_manifest.py` lightly: it reshuffles train/test with a fixed seed over case order. Treat `cases.csv` as frozen unless you intentionally cut a new split version.

`data/labels/chains.jsonl` is **gitignored** (cluster artifact). Its `split` field must match `cases.csv` via:

```bash
uv run python -m scripts.data.restamp_chains_splits --dry-run
uv run python -m scripts.data.restamp_chains_splits --backup data/labels/chains.jsonl.bak
```

After restamping, rebuild HybridRAG / HippoRAG indexes (`FORCE_REBUILD=1` for HybridRAG). Coordinate before wiping the shared Chroma path in `configs/paths.yaml` → `rag.chroma_db_storage`.

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
| `split` | `train` / `test` — **split authority** for RAG, chains, LoRA, eval |
| `n_slides` | Slide count |

```bash
# Only when intentionally creating a new split version:
uv run python -m scripts.data.build_manifest
```

Example: `cases.csv.example`
