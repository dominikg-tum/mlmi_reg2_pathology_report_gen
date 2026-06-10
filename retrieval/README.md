# Retrieval (DOMI)

Phases **1, 2, 4** only (no Phase 3 / ABMIL).

| Phase | Capability |
|-------|------------|
| **1** | CONCHv1.5 at **5×/10×/20×/40×** (variable native patch px → 224 CONCH input) → `patch_embeddings_{zoom}.pt`; CONCH text×vision cosine; `node.zoom_level` + `description` |
| **2** | Adjacent-scale ×10 parent for each ×20 patch (geometry via `coords_medium.pt`) |
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

`graph_guided` passes `node.zoom_level` → `embeddings_{band}.pt` (separate pool per magnification; not a flat pool).
