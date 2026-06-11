# Vision (DOMI)

Patch tiling uses **512×512 px** at ×4 / ×10 / ×20 (`configs/vision.yaml`). CONCHv1.5 via `TitanEncoder.return_conch()` only.

## P2 — patch retrieval pipeline

| Step | Script | Output |
|------|--------|--------|
| Tile | `scripts/vision/tile_slides.py` | `coords_{medium,high}.pt`, `meta_*.json` |
| Verify | `scripts/vision/verify_tiling.py` | `tiling_preview_high.png`, `tiling_verified.flag` |
| Encode | `scripts/vision/encode_patches_offline.py` | `embeddings_{medium,high}.pt` |
| K-means | `scripts/vision/build_kmeans_index.py` | `kmeans_centroids_{medium,high}.pt` |
| Demo | `scripts/vision/run_retrieval_demo.py` | `demo_<stem>.png` |

Per slide cache layout:

```text
{cache_root}/{slide_id}/
  coords_medium.pt
  coords_high.pt
  embeddings_medium.pt
  embeddings_high.pt
  kmeans_centroids_medium.pt
  kmeans_centroids_high.pt
  meta_medium.json
  meta_high.json
  tiling_preview_high.png
```

## P1 baselines (unchanged)

| Baseline | `--visual` | Offline job |
|----------|------------|-------------|
| Thumbnail | `thumbnail` | `build_thumbnail_cache.py` |
| TITAN slide embed | `slide_embed` | `encode_slide_embeddings.py` |

### Team thumbnail banks (`/mnt/projects/mlmi/reg2/dataset/`)

Precomputed JPEGs (460 slides) — use while full CONCH/TITAN offline encode is in flight:

| Directory | Use |
|-----------|-----|
| `thumbnails/` | Default pyramid downsample |
| `thumbnails_kmeans/` | Tissue-emphasized (k≈100) — recommended over plain thumbnails |
| `thumbnails_kmeans_5/` | Coarse tissue summary (k=5) — ablation |

Select bank via `configs/vision.yaml` → `thumbnail.variant`. See [PROJECT_OVERVIEW.md §2a](../docs/PROJECT_OVERVIEW.md#2a-thumbnail-options-cluster).

Data WSIs: `/mnt/projects/mlmi/reg2/TUMUntera` (see `configs/paths.yaml`).
