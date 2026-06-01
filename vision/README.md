# Vision (DOMI)

## P1 — thumbnail (no TITAN)

```bash
python scripts/vision/build_thumbnail_cache.py
python -m baselines.run_agent --visual thumbnail
```

## P2 — offline encode then retrieve

```bash
python scripts/vision/encode_patches_offline.py   # sbatch on cluster
python -m baselines.run_agent --visual patch_retrieve --retriever titan_cosine
```

Cache layout: `configs/vision.yaml` → `cache_root` (cluster default: `dominik/cache` via `paths.yaml`) → `{slide_id}/thumbnail.png`, `embeddings_high.pt`, etc.

## MMNavAgent

Future: `vision/navigation.py` → `get_navigator("mnavagent")` when code is public.
