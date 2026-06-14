# WSI Patch Selection & Retrieval — Research Comparison

*For: Agentic Pathology Report Generation via VLMs (UTERUS graph + WSI)*

**See also:** [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for the locked target architecture and codebase audit. [WSI_Background_Filtering.md](WSI_Background_Filtering.md) for tissue vs glass filtering during offline tiling.

---

## 0. Context & Constraints

| Dimension | Value |
|-----------|-------|
| Dataset | ~220 WSIs (150 train+val / 70 test) under `/mnt/projects/mlmi/TUMUntera/TUM_Untera_data` |
| Offline backbone | CONCHv1.5 (`MahmoodLab/CONCH`, 768-d) + TITAN slide encoder (`MahmoodLab/TITAN`, 1024-d) |
| Tiling (locked) | **Four zoom levels**, non-overlapping native patch sizes → resize to **224×224** before CONCH encode |
| Agent workflow | Graph Q&A → **`node.zoom_level`** → load `patch_embeddings_{tier}.pt` → **K-means centroid pool (default)** → `TitanEncoder.encode_text()` cosine → raw patches → VLM |
| Report stage | **MedGemma 1.5 4B** chain-only baseline today; optional slide context via native images or trained TITAN adapter (Phase 2) — see [PROJECT_OVERVIEW.md §2e](PROJECT_OVERVIEW.md#2e-images-vs-embeddings--when-to-use-what) |
| Thumbnail baseline | **Done on cluster** — `/mnt/projects/mlmi/reg2/dataset/thumbnails/` |
| Hardware | 1× A100-80G or 2× A100-40G |

---

## 1. Patch Size & Zoom Level

### Primary path — `zoom_level` on every graph node

Each node carries **`zoom_level`**: `"5x"`, `"10x"`, `"20x"`, or `"40x"`. At inference the agent loads **`patch_embeddings_{tier}.pt`** for that WSI — never a single flat pool.

| `zoom_level` | Native tile (non-overlap) | CONCH input | Artifact | Typical use | top-k |
|--------------|---------------------------|-------------|----------|-------------|-------|
| **`5x`** | 2048×2048 | 224×224 | `patch_embeddings_5x.pt` | Global architecture | **3** |
| **`10x`** | 1024×1024 | 224×224 | `patch_embeddings_10x.pt` | Compartment, gross pattern | **3** |
| **`20x`** | 512×512 | 224×224 | `patch_embeddings_20x.pt` | Gland/stroma detail; **TITAN source** | **5** |
| **`40x`** | 256×256 | 224×224 | `patch_embeddings_40x.pt` | Nuclear detail, mitoses | **5** |

**VLM gets raw patch images**, not CONCH embeddings. Embeddings are retrieval-only.

### TITAN slide embedding — ×20 only (Phase 2)

TITAN was trained on **512×512 @ 20×**. `slide_embedding.pt` (1024-d) aggregates the ×20 pool only. Multi-scale CONCH pools are for Phase 1 retrieval.

### Flat ×20 pool (ablation)

Force all nodes to `patch_embeddings_20x.pt` regardless of `zoom_level`.

---

## 2. Phase 1 retrieval (locked steps)

```
a. tier = node["zoom_level"]
   patch_embeddings = load("patch_embeddings_{tier}.pt")

b. node_text_emb = TitanEncoder.encode_text(node.retrieval_text)  # question + description, (768,)

c. # Default: rank K-means centroids (kmeans_k=100 in configs/vision.yaml), then load winning patches
   # Optional: search full patch pool (--search-all-patches) — simpler, slower
   similarities = cosine_similarity(node_text_emb, centroid_embeddings[tier])
   top_k_indices = argsort(similarities, descending=True)[:k]
   top_k_patches = load_raw_patches(tier, top_k_indices)   # raw images for VLM

d. answer = VLM(top_k_patches, node.question, HippoRAG_context)
   # current default: Qwen3-VL-8B-Instruct; ablation: InternVL3.5-8B
```

Configured in `configs/vision.yaml` → `retrieval.top_k_by_zoom`, `retrieval.kmeans_k`, `retrieval.adjacent_scale`. Implemented via `GraphGuidedRetriever` + `node.retrieval_text` + `PatchRetrieveProvider`. Encoder decisions: [PROJECT_OVERVIEW.md §2b](PROJECT_OVERVIEW.md#2b-locked-architectural-decisions).

### Adjacent-scale enrichment (CMT)

When `retrieval.adjacent_scale.enabled: true`, each retrieved patch also loads its **parent** tile from the next-coarser pool (`parent_map` in config). Integration/report nodes (`tier=integration` or `node_kind=integration|report`) additionally load a **grandparent**. Images are passed to the VLM alongside the primary patch — embedding-level fusion is not used.

### HippoRAG 2 (semantic memory, not patch retrieval)

Retrieves similar past CoT steps given `(node_id + partial chain)`. Top-2 steps as text context. Online update after each node.

---

## 3. Offline preprocessing (add to existing)

```
WSI (.svs) under /mnt/projects/mlmi/TUMUntera/TUM_Untera_data
  → thumbnail (done on cluster)
  → tile at 5×/10×/20×/40× (native px per table above)
  → resize each patch to 224×224 → CONCHv1.5 vision encode via TitanEncoder.return_conch()
  → save patch_embeddings_{5x,10x,20x,40x}.pt  (N × 768 each)
  → K-means centroids per pool (default k=100) → kmeans_centroids_{zoom}.pt
  → TITAN slide encode on 20× pool only → slide_embedding.pt
```

Cluster scripts: `scripts/vision/tile_slides.py`, `encode_patches_offline.py`, `build_kmeans_index.py`, `encode_slide_embeddings.py`

Tissue vs glass: current rule is grayscale mean ≤ 220 (`vision/wsi_io.py`). Recommended upgrade — slide-level HSV mask + minimum tissue fraction — documented in [WSI_Background_Filtering.md](WSI_Background_Filtering.md).

---

## 4. Method Overview Matrix

| # | Method | Zoom-aware? | Graph-aware? | Status |
|---|--------|-------------|--------------|--------|
| A | **CONCH cross-modal cosine** | Yes (4 pools) | Query + `description` via `encode_text()` | **Primary** — `TitanEncoder` single load |
| B | **`zoom_level` routing** | Yes | Yes | **Primary path** |
| C | **K-means centroid pool** | Yes | — | **Default on** (`kmeans_k=100`; optional full-pool search) |
| D | **MMNavAgent CMT parent** | Yes | Partial | Config `parent_map`: 40×→20×, 20×→10×, 10×→5×; grandparent on integration nodes |

---

## 5. Key numbers

| Quantity | Value |
|----------|-------|
| CONCH patch embedding dim | 768 |
| CONCH pools per WSI | 4 (`5x`, `10x`, `20x`, `40x`) |
| CONCH vision input | 224×224 (from variable native tile) |
| TITAN slide embedding dim | 1024 |
| TITAN source zoom | 20× only |
| top-k @ 5× / 10× | 3 |
| top-k @ 20× / 40× | 5 |
| K-means centroids per pool | 100 (default; tune in `configs/vision.yaml`) |
| HippoRAG retrieval k | 2 CoT steps |

---

## 6. References

| Paper | Link |
|-------|------|
| PathChat+ & SlideSeek | [arXiv:2506.20964](https://arxiv.org/abs/2506.20964) |
| MMNavAgent | [arXiv:2603.02079](https://arxiv.org/abs/2603.02079) |
| TITAN / CONCH | MahmoodLab |
| HippoRAG 2 | github.com/OSU-NLP-Group/HippoRAG |
