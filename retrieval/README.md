# Retrieval (DOMI)

Phases **1, 2, 4** only (no Phase 3 / ABMIL).

| Phase | Capability |
|-------|------------|
| **1** | CONCHv1.5 at **20× only** → `patch_embeddings_20x.pt`; CONCH text×vision cosine over fixed pool; `node.zoom_level` is an ontology hint only |
| **2** | Adjacent-scale parents are disabled by default; coarse context is provided by attaching the whole-slide thumbnail |
| **4** | Spatial diversity filter (`d_min_20x_px` from `configs/vision.yaml`) |

## Offline pipeline (cluster order)

```bash
# 1. Tile (CPU)
sbatch scripts/cluster/tile_slides.sh

# 2. Visual gate — one slide
python -m scripts.vision.verify_tiling --wsi-path /mnt/projects/mlmi/reg2/TUMUntera/<slide>.svs

# 3. Encode (GPU) — after tiling_verified.flag exists
sbatch scripts/cluster/encode_patches.sh

# 4. K-means (CPU)
sbatch scripts/cluster/build_kmeans.sh

# 5. Smoke test
python -m scripts.vision.run_retrieval_demo \
  --wsi-path ... --question "irregular crowded endometrial glands" \
  --node-tier local_features --k 5
```

## Agent

```bash
python -m baselines.run_agent \
  --visual patch_retrieve \
  --retriever graph_guided \
  --slide-id YOUR.svs
```

`graph_guided` uses `retrieval.fixed_pool` (default `20x`) and loads `patch_embeddings_20x.pt` regardless of `node.zoom_level`.
