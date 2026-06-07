# WSI Patch Selection & Retrieval — Research Comparison
*For: Agentic Pathology Report Generation via VLMs (UTERUS graph + WSI)*

---

## 0. Context & Constraints

| Dimension | Value |
|-----------|-------|
| Dataset size | ~220 WSIs (150 train+val / 70 test) |
| Backbone | TITAN embeddings (CONCHv1.5 patch encoder, 512×512 px @ ×20, 768-dim features) |
| Agent workflow | Graph-guided Q&A → VLM answers from retrieved patches + episodic/semantic memory |
| Baseline already done | Thumbnail (single low-res view) |
| Target | Multi-patch, multi-zoom retrieval conditioned on **graph node / question context** |

---

## 1. Patch Size & Zoom Level — What the Data Says

### Standard choices in SOTA
| Magnification | Pixel size on slide | Typical patch px | What it shows |
|---------------|--------------------|--------------------|---------------|
| ×4 | ~2.5 µm/px | 512–1024 px | Global architecture, whole-tissue organisation |
| ×10 | ~1 µm/px | 512 px | Glandular/stromal patterns |
| **×20** | **~0.5 µm/px** | **512×512 px** | **Cellular morphology — TITAN default** |
| ×40 | ~0.25 µm/px | 256–512 px | Nuclear detail, mitoses |

**TITAN specifically** tiles WSIs at **×20, non-overlapping 512×512 px** and encodes each with CONCHv1.5 into a 768-dim vector before slide-level iBOT pretraining. This is therefore the "native" scale for TITAN-based retrieval.

**For the uterus domain:**
- Global questions (e.g., "Is the endometrium atrophic overall?") → ×4–×10
- Local tissue questions (e.g., "Are glands irregular?") → ×10–×20
- Integration/detail questions (e.g., "Are there mitotic figures?") → ×20–×40

This maps naturally onto your **UTERUS graph tier structure**.

---

## 2. Method Overview Matrix

| # | Method | Supervised? | Zoom-aware? | Graph-aware? | Complexity | Best for |
|---|--------|-------------|-------------|--------------|------------|----------|
| A | **TITAN + Cosine Sim** (Domi initial) | No | Fixed (×20) | Query-aware | Low–Med | Semantic retrieval baseline |
| B | **UTERUS graph-tier zoom** (Domi advanced) | No | **Hierarchical (graph-driven)** | **Yes, natively** | Med | Your project's target architecture |
| C | **MMNavAgent** (TUM, 2026) | Yes (slide label) | Adaptive multi-mag | No (post-hoc) | High | Standalone diag. classification |
| D | **PathFinder** (ICCV 2025) | Partial | Multi-scale | Partially | High | Report generation inspiration |
| E | **Attention map top-k** (ABMIL/CLAM) | Weakly (slide label) | Fixed | No | Med | Discriminative patch mining |
| F | **K-means clustering** | No | Fixed | No | Low | Coverage/diversity baseline |
| G | **Random selection** | No | Fixed | No | Trivial | Ablation baseline only |
| H | **SSL contrastive pretraining** | No | Fixed | No | High (training) | Feature quality, not retrieval |

---

## 3. Method Deep-Dives

---

### 3A. TITAN Embedding + Cosine Similarity *(Domi's Initial Idea)*

**How it works:**
1. **Offline preprocessing**: tile all WSIs at ×20 (512×512 px), run CONCHv1.5 encoder → store patch feature matrix `F ∈ ℝ^{N×768}` per slide.
2. **Inference**: for each graph node question `q`, encode `q` with TITAN's text encoder → `q_emb ∈ ℝ^{768}`.
3. Compute cosine similarity `sim(q_emb, F_i)` for all N patches.
4. Return top-k patches (e.g., k=3–10) as context for the VLM.

**Zoom level**: Fixed at ×20 unless explicitly multi-scale.

**Pros:**
- Directly exploits TITAN's **joint vision-language embedding space** — text and image are aligned.
- Zero extra training (works at inference time out of the box).
- Fast: `O(N)` dot products per query; precomputed patch embeddings.
- Naturally **question-conditioned** — different graph nodes pull different patches.

**Cons:**
- Single zoom level: misses global architectural context (glandular arrangement only visible at ×10).
- Top-k may cluster spatially (all similar regions); no diversity guarantee.
- TITAN was trained on diverse cancers (Mass General Brigham), uterus-specific alignment may be weaker.
- No awareness of graph structure (treats each node question independently).

**Recommended k**: 3–8 patches per graph node query. Start with k=5.

**Implementation note**: TITAN provides `encode_text()` for the query side; patch features can be extracted once offline.

---

### 3B. UTERUS Graph-Tier Zoom *(Domi's Advanced Idea — Recommended Core)*

**How it works:**
Map graph tiers to zoom levels, then run TITAN cosine sim retrieval at the appropriate scale:

```
Graph tier       →   Zoom level    →   Patch size    →   What it captures
─────────────────────────────────────────────────────────────────────────
Global node      →   ×4–×10        →   512×512 px   →   Overall tissue architecture
Local node       →   ×10–×20       →   512×512 px   →   Glandular/stromal patterns  
Integration node →   ×20–×40       →   512×512 px   →   Cellular / mitotic detail
```

**Detailed pipeline:**
1. **Preprocess multi-scale**: tile at ×4, ×10, ×20 (and optionally ×40). Store TITAN/CONCH embeddings for each scale separately.
2. **At inference**: graph walks a node → read its tier label → select the appropriate zoom table.
3. Run TITAN cosine sim at that zoom level → top-k patches.
4. Optionally: for integration nodes that span scales, run on ×10 **and** ×20, union the top-k (deduplicate by spatial overlap).

**Extension — spatial diversity**: After top-k retrieval, apply a **minimum spatial distance filter** so returned patches are not all from the same tissue region. Enforcing `d_min` between patch centres in WSI coordinates ensures coverage.

**Why this is the right architectural choice for your project:**
- Your graph is already doing the reasoning about *what kind of question* is being asked.
- The graph tier directly encodes the **granularity of evidence needed** — this is domain knowledge that MMNavAgent has to learn from data, but you have it for free.
- Keeps retrieval interpretable and auditable: a pathologist can see "this node was answered from ×20 patches because it is a local-detail node."
- Minimal additional infrastructure: reuse TITAN embeddings, add a zoom routing table.

**Cons:**
- Requires extracting and storing patches at 3 zoom levels (~3× storage).
- Tier labelling of graph nodes needs to be explicit and consistent.
- No cross-scale aggregation (a single patch doesn't see context from adjacent zoom levels — see MMNavAgent for that).

---

### 3C. MMNavAgent *(TUM, arXiv 2603.02079, March 2026)*

**How it works:**
Two components in a closed loop:
- **Cross-Magnification Navigation Tool (CMT)**: at a chosen magnification, generates attention heatmaps to extract high-attention regions, aggregating features from *adjacent* magnification levels (lower mag for context, higher for detail).
- **Magnification Selection Tool (MST)**: an LLM-based agent that starts from the thumbnail, uses a memory bank of prior navigation steps, and decides the next action: `zoom_in`, `zoom_out`, `move`, or `stop`.

**Loop**: thumbnail → MST selects initial mag → CMT navigates at that mag → top regions stored in memory → MST reads memory → picks next action → repeat until `stop`.

**Results**: +1.45% AUC, +2.93% BACC over non-agent baseline on public dataset.

**Relevance to your project:**
- The MST/CMT loop is the closest existing work to your "different graph tiers → different zoom" idea, but driven by learned policy rather than graph structure.
- CMT's cross-magnification feature aggregation (adjacent-scale fusion) is directly applicable: when you retrieve at ×20, you could also pull the ×10 context of the same region and concatenate.
- **Best adoption strategy**: use the **CMT concept** (adjacent-scale feature fusion) within your graph-tier routing. When a node query fires at ×20, also sample the ×10 parent patch of each retrieved patch and concatenate — richer context at minimal cost.

**Cons for direct use:**
- Requires training the MST policy (slide-level labels needed).
- Adds significant complexity; overkill if the graph already encodes zoom routing.
- Code "will be public upon acceptance" — not yet available as of June 2026.

---

### 3D. PathFinder *(ICCV 2025, arXiv 2502.08916)*

**How it works:**
Four collaborating agents:
1. **Triage Agent**: classifies WSI as benign/risky → skips navigation if benign.
2. **Navigation Agent**: generates importance maps over the WSI → identifies diagnostically significant regions.
3. **Description Agent**: VLM describes sampled patches in natural language.
4. **Diagnosis Agent**: synthesizes patch descriptions → final diagnosis.

**Results**: 74% accuracy on M-Path skin melanoma dataset; surpasses average pathologist by 9%, SOTA by 8%.

**Relevance to your project:**
- Very similar **spirit** to yours (iterative evidence gathering + natural language synthesis).
- The Navigation Agent's importance map + sampling loop is directly analogous to your patch retrieval step.
- Key difference: PathFinder has no graph structure — your UTERUS graph provides explicit reasoning scaffolding that PathFinder lacks.
- **Description Agent** is a good model for your VLM answering step: ask the VLM to describe retrieved patches before answering the graph-node question.

**What to borrow**: the iterative "navigate → describe → accumulate → synthesize" loop as a blueprint for your VLM answering chain.

---

### 3E. Attention Map Top-k *(ABMIL / CLAM)*

**How it works:**
1. Train an attention-based MIL model (ABMIL or CLAM) on your 150 training WSIs with slide-level labels (e.g., diagnosis category).
2. At inference: run forward pass → extract per-patch attention scores `α_i`.
3. Return top-k patches by attention score as the "most diagnostic" patches.

**Known issue**: Top-10 attention scores in ABMIL can account for >85% of total attention mass — the model over-concentrates on very few patches and misses diversity. Stochastic top-K masking (ACMIL) was proposed to counteract this.

**Relevance to your project:**
- Works well as a *supervised* complement to TITAN cosine sim: train once on your 150 slides, then use the attention map for any question.
- Limitation: attention is conditioned on slide-level label, not on graph-node question → the same patches are selected regardless of which question is being answered.
- Best used as a **pre-filter**: use attention map to identify the top 20% of patches per slide as a candidate pool, then run TITAN cosine sim within that pool for question-specific retrieval.

**Two-stage pipeline** (recommended hybrid):
```
ABMIL attention → top-20% candidate pool  →  TITAN cosine sim (within pool)  →  top-k patches
```
This reduces the cosine sim search space by 5×, cuts noise, and is still question-conditioned.

---

### 3F. K-means Clustering *(unsupervised, easy baseline)*

**How it works:**
1. Extract CNN/CONCH features for all patches.
2. Cluster into K groups (K = 10–50 depending on slide size).
3. Select the centroid patch (or a few patches per cluster) as representatives.

**Empirical result**: K-means reduces storage and search by 50–90% with only a few % accuracy loss vs. using all patches (Springer 2019, Overcoming Patch-Based Limitations 2020).

**Relevance to your project:**
- Good **preprocessing step**: run K-means once per WSI to reduce the patch library from ~10k to ~200–500 representative patches. Then run TITAN cosine sim only on those.
- Captures tissue *diversity* across the slide (different morphological zones).
- Does not condition on question — purely coverage-based.

**Practical recommendation**: use K=50–100 per slide. At ~220 WSIs this is very feasible. Combine with TITAN cosine sim for question-conditioned retrieval on top.

---

### 3G. Random Selection *(trivial baseline)*

Select N patches uniformly at random from non-background tissue.
- Good for sanity-check ablation ("how much does retrieval matter?").
- Often surprisingly competitive for global questions where any tissue patch is informative.
- Implement as your lowest-effort baseline.

---

### 3H. Self-Supervised Contrastive Learning *(indirect relevance)*

Frameworks like SimCLR, DINO, iBOT trained on your specific uterus dataset could produce domain-adapted patch embeddings that outperform CONCHv1.5 for retrieval. However:
- Your 150 training slides are likely **too few** for training a new patch encoder from scratch.
- CONCHv1.5/TITAN already provides strong pathology priors.
- **Practical use**: fine-tune CONCHv1.5 with a contrastive objective on your uterus patches (optional, high effort, last resort if TITAN cosine sim retrieval performs poorly).

---

## 4. Recommended Architecture for Your Project

### Phase 1 — MVP (implement first)

```
Preprocessing (offline, once):
  WSI  →  tile at ×20 (512×512 px)  →  CONCHv1.5 encode  →  store patch_embs_20x.npy
         tile at ×10 (512×512 px)  →  CONCHv1.5 encode  →  store patch_embs_10x.npy

Inference (per graph node):
  graph_node  →  read tier {global, local, integration}
              →  select zoom table {×10, ×20, ×20+×40}
              →  encode node question  →  q_emb  (TITAN text encoder)
              →  cosine_sim(q_emb, patch_embs_Zx)
              →  top-k patches  →  VLM context
```

**Concrete zoom routing table (starting point):**
| Graph Tier | Zoom | Rationale |
|------------|------|-----------|
| Global (e.g. overall architecture, invasion depth) | ×10 | Architecture visible |
| Local (gland morphology, stroma) | ×20 | TITAN native |
| Integration (mitoses, cytological atypia) | ×20 | Cellular detail |
| Highly detailed (nuclear features) | ×40 | Optional, add last |

### Phase 2 — Add Adjacent-Scale Context (CMT-inspired)

For each top-k patch retrieved at ×20, also include its **parent patch at ×10** (the ×10 region containing the ×20 patch) in the VLM context. This gives the VLM both local cellular detail and regional context — mimicking the CMT cross-magnification fusion in MMNavAgent, but without needing to train the MST policy.

### Phase 3 — Attention-Based Pre-filtering (if compute allows)

Train a lightweight ABMIL on your 150 training slides → generate attention maps for the 70 test slides at inference → use top-20% as candidate pool for TITAN cosine sim retrieval.

### Phase 4 — Spatial Diversity Enforcement

After retrieving top-k by cosine sim, apply a minimum spatial distance filter (`d_min = ~512 px at WSI coordinates`) so all k patches come from spatially distinct tissue regions.

---

## 5. Quick Comparison Table — Fit for Your Use Case

| Method | Needs training? | Question-conditioned? | Multi-zoom? | Uterus-specific? | Effort | Recommend? |
|--------|----------------|----------------------|-------------|-----------------|--------|------------|
| TITAN cosine sim (3A) | ❌ | ✅ | ❌ (fix) | Partial | Low | ✅ Start here |
| Graph-tier zoom (3B) | ❌ | ✅ | ✅ | ✅ | Med | ✅ **Core method** |
| Adjacent-scale fusion (CMT idea) | ❌ | ✅ | ✅ | ✅ | Med | ✅ Phase 2 |
| ABMIL attention pre-filter (3E) | ✅ (slide labels) | ❌ (pre-filter only) | ❌ | ✅ | Med | ✅ Phase 3 |
| K-means preprocessing (3F) | ❌ | ❌ (pre-filter) | ❌ | ✅ | Low | ✅ Phase 1 preprocessor |
| MMNavAgent full (3C) | ✅ (heavy) | Partial | ✅ | ❌ | Very High | ❌ Borrow CMT concept only |
| PathFinder full (3D) | ✅ | Partial | ✅ | ❌ | High | ❌ Borrow loop concept |
| Random (3G) | ❌ | ❌ | ❌ | ❌ | Trivial | ✅ Ablation baseline |

---

## 6. Key Numbers to Know

- **TITAN default patch size**: 512×512 px at ×20 magnification
- **Typical uterine WSI at ×20**: ~5,000–15,000 tissue patches
- **After K-means (K=100)**: ~100 representative patches per slide
- **ABMIL attention concentration**: top-10 patches > 85% attention mass (ECCV 2024)
- **K-means accuracy loss**: 5–10% retrieval accuracy for 50–90% storage reduction (Springer 2019)
- **MMNavAgent gain**: +1.45% AUC, +2.93% BACC over non-agent baseline (TUM 2026)
- **PathFinder**: 74% accuracy, +9% over average pathologist on melanoma dataset (ICCV 2025)
- **TITAN training**: 335,645 WSIs + 182,000 pathology reports + 423,122 synthetic captions

---

## 7. Open Questions / Next Experiments

1. **Tier labelling**: Is every node in the UTERUS graph explicitly tagged as global / local / integration? If not, assign this first — it drives the entire zoom routing.
2. **k value sweep**: Compare k = {1, 3, 5, 10} patches per node on validation set. Likely diminishing returns after k=5–8.
3. **Zoom validation**: Does retrieving ×10 patches for global nodes actually improve VLM answers vs. ×20-only? Run ablation on 10–15 validation slides.
4. **TITAN text encoder quality on uterus**: check if standard pathology terms ("endometrial glands", "stromal decidualisation", etc.) produce sensible cosine rankings before committing to the full pipeline.
5. **LoRA fine-tuning**: if retrieval quality is good but VLM answering is weak, LoRA on the VLM backbone with (patch, graph_question, answer) triples is the next lever.

---

## 8. References

| Paper | Key contribution | Link |
|-------|-----------------|------|
| TITAN (Nature Medicine 2025) | Multimodal WSI foundation model; 512×512 @ ×20; vision-language aligned | github.com/mahmoodlab/TITAN |
| MMNavAgent (arXiv 2603.02079, TUM 2026) | MST + CMT loop for adaptive multi-magnification navigation | arxiv.org/abs/2603.02079 |
| PathFinder (ICCV 2025, arXiv 2502.08916) | 4-agent VLM pipeline; triage → navigate → describe → diagnose | arxiv.org/abs/2502.08916 |
| ABMIL (Ilse et al. 2018) | Attention-based MIL; attention scores as patch importance | — |
| CLAM (Lu et al. 2021) | Clustering-constrained attention MIL; top-k patch mining | — |
| ACMIL / ECCV 2024 | Stochastic top-K masking; top-10 > 85% attention mass finding | — |
| Patch clustering (Springer 2019) | SOM + GMM for WSI representation; 50% patches = ~65% accuracy | link.springer.com/chapter/10.1007/978-3-030-23937-4_4 |
| Overcoming patch-based limits (2020) | Mini-batch K-means on 1.4M patches; coverage vs. random | arxiv.org/abs/2012.00617 |
| HIPT / SSL scaling (medRxiv 2023) | iBOT ViT pretraining on 40M histology patches; SOTA SSL baseline | medrxiv.org/content/10.1101/2023.07.21.23292757 |
| Fast patch selection FPS (2023) | KDE-based spatial density sampling; diversity + coverage | arxiv.org/abs/2311.08359 |
| FAST imaging Python WSI tutorial | Practical FAST library for patch extraction | fast-imaging.github.io/python-tutorial-wsi.html |
