# WSI Patch Selection & Retrieval — Research Comparison

*For: Agentic Pathology Report Generation via VLMs (UTERUS graph + WSI)*

**See also:** [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for the locked target architecture and codebase audit. [WSI_Background_Filtering.md](WSI_Background_Filtering.md) for tissue vs glass filtering during offline tiling.

---

## 0. Context & Constraints

| Dimension | Value |
|-----------|-------|
| Dataset | ~220 WSIs (150 train+val / 70 test) under `/mnt/projects/mlmi/TUMUntera/TUM_Untera_data` |
| Offline backbone | CONCHv1.5 (`MahmoodLab/CONCH`, 768-d) + TITAN slide encoder (`MahmoodLab/TITAN`, 1024-d) |
| Tiling (locked) | **20× only** offline patch pool (512×512 native) → resize to **224×224** before CONCH encode |
| Agent workflow | Graph Q&A → **fixed 20× retrieval pool** (`patch_embeddings_20x.pt`) → `TitanEncoder.encode_text()` cosine → raw patches (+ thumbnail) → VLM; `node.zoom_level` is an ontology hint only |
| Report stage | **MedGemma 1.5 4B** chain-only baseline today; optional slide context via native images or trained TITAN adapter (Phase 2) — see [PROJECT_OVERVIEW.md §2e](PROJECT_OVERVIEW.md#2e-images-vs-embeddings--when-to-use-what) |
| Thumbnail baseline | **Done on cluster** — `/mnt/projects/mlmi/reg2/dataset/thumbnails/` |
| Hardware | 1× A100-80G or 2× A100-40G |

---

## 1. Patch Size & Zoom Level

### Primary path — fixed 20× retrieval pool

At inference, **all patch-retrieve nodes** rank the same offline pool: `patch_embeddings_20x.pt`. The graph still stores `zoom_level` for readability and prompt hints, but it does not route caches.

| `zoom_level` | Native tile (non-overlap) | CONCH input | Artifact | Typical use | top-k |
|--------------|---------------------------|-------------|----------|-------------|-------|
| **`thumbnail`** | — | — | team JPEG bank | Global architecture | — |
| **`20x`** | 512×512 | 224×224 | `patch_embeddings_20x.pt` | Retrieval pool; **TITAN source** | **5** |
| **runtime 10x/20x/40x** | 1024 / 512 / 256 | — | none | Pixel crops for ReAct zoom | — |

**VLM gets raw patch images**, not CONCH embeddings. Embeddings are retrieval-only.

### TITAN slide embedding — ×20 only (Phase 2)

TITAN was trained on **512×512 @ 20×**. `slide_embedding.pt` (1024-d) aggregates the ×20 pool only. Multi-scale CONCH pools are for Phase 1 retrieval.

### Runtime zoom (ReAct)

When evidence is insufficient, the agent may either re-retrieve (new `sub_query`) or request a runtime crop at `10x`, `20x`, or `40x` around the best retrieved 20× coordinate. This uses `openslide` reads only and does not require offline 10×/40× embedding pools.

---

## 2. Phase 1 retrieval (locked steps)

```
a. pool = "20x"
   patch_embeddings = load("patch_embeddings_20x.pt")

b. node_text_emb = TitanEncoder.encode_text(node.retrieval_text)  # question + description, (768,)

c. # Default: rank full pool (search_all_patches: true). Optional K-means centroid restriction is ablation-only.
   similarities = cosine_similarity(node_text_emb, patch_embeddings)
   top_k_indices = argsort(similarities, descending=True)[:k]
   top_k_patches = load_raw_patches(tier, top_k_indices)   # raw images for VLM

d. answer = VLM(top_k_patches, node.question, HippoRAG_context)
   # current default: Qwen3-VL-8B-Instruct; ablation: InternVL3.5-8B
```

Configured in `configs/vision.yaml` → `retrieval.top_k_by_zoom`, `retrieval.kmeans_k`, `retrieval.adjacent_scale`. Implemented via `GraphGuidedRetriever` + `node.retrieval_text` + `PatchRetrieveProvider`. Encoder decisions: [PROJECT_OVERVIEW.md §2b](PROJECT_OVERVIEW.md#2b-locked-architectural-decisions).

### Adjacent-scale enrichment (CMT)

Disabled by default in this repo. Coarse context is provided by always attaching the whole-slide thumbnail alongside the retrieved 20× patches.

### HippoRAG 2 (semantic memory, not patch retrieval)

Retrieves similar past CoT steps given `(node_id + partial chain)`. Top-2 steps as text context. Online update after each node.

---

## 3. Offline preprocessing (add to existing)

```
WSI (.svs) under /mnt/projects/mlmi/TUMUntera/TUM_Untera_data
  → thumbnail (done on cluster)
  → tile at 20× only (native 512 px)
  → resize each patch to 224×224 → CONCHv1.5 vision encode via TitanEncoder.return_conch()
  → save patch_embeddings_20x.pt  (N × 768)
  → Optional K-means (ablation) → kmeans_centroids_20x.pt
  → TITAN slide encode on 20× pool only → slide_embedding.pt
```

Cluster scripts: `scripts/vision/tile_slides.py`, `encode_patches_offline.py`, `build_kmeans_index.py`, `encode_slide_embeddings.py`

Tissue vs glass: current rule is grayscale mean ≤ 220 (`vision/wsi_io.py`). Recommended upgrade — slide-level HSV mask + minimum tissue fraction — documented in [WSI_Background_Filtering.md](WSI_Background_Filtering.md).

---

## 4. Method Overview Matrix

| # | Method | Zoom-aware? | Graph-aware? | Status |
|---|--------|-------------|--------------|--------|
| A | **CONCH cross-modal cosine** | Single pool (20×) | Query + `description` via `encode_text()` | **Primary** — `TitanEncoder` single load |
| B | **Fixed 20× pool routing** | No | Yes | **Primary path** |
| C | **K-means centroid pool** | No | — | Ablation only |
| D | **MMNavAgent CMT parent** | No | Partial | Not used (thumbnail replaces parent tiles) |

---

## 5. Key numbers

| Quantity | Value |
|----------|-------|
| CONCH patch embedding dim | 768 |
| CONCH pools per WSI | 1 (`20x`) |
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
