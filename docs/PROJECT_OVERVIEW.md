# MLMI REG² — Interactive Pathology Report Generation

**TUM MLMI Practical Course · Summer 2026** · Dr. Han Li · REG² challenge-oriented project

| Also read | |
|-----------|---|
| [cluster_setup.md](cluster_setup.md) | Garching SLURM, enroot, vLLM |
| [WSI_Patching_Retrieval_Research.md](WSI_Patching_Retrieval_Research.md) | Patch retrieval methods & zoom routing |
| [../README.md](../README.md) | Repo tree, quick start, owner lanes |

---

## 1. Goal

Build a system that, **given only a WSI at test time**, walks a **diagnostic graph** question-by-question, answers from **visual evidence** (+ memory), and outputs:

1. **Reasoning chain** — evaluated with Binary Path Validity, Edge-F1, MESS  
2. **Final pathology report** — ROUGE-L, BLEU-4, clinical accuracy  

**Train vs inference (critical):**

| | Training | Inference |
|---|----------|-----------|
| Input | WSI + report | **WSI only** |
| Report | Supervision (WP3 chains) + HippoRAG 2 index (train split only) | **Not available** |

**WSI data:** `.svs` files under `/mnt/projects/mlmi/TUMUntera/TUM_Untera_data` (~220 slides).

**Cluster-only assets** (not on local laptops — see [cluster_setup.md](cluster_setup.md)):

| Location | Contents |
|----------|----------|
| `/mnt/projects/mlmi/reg2/containers/` | Team `.sqsh` bases: `qwen25_graphrag.sqsh`, `qwen25.sqsh`, `qwen25_dev_updated.sqsh`, `qwen25_dev_v2.sqsh`; personal exports e.g. `dominik_20260529_base.sqsh`. **Each person creates their own** enroot env + export — follow §3 in `cluster_setup.md`. |
| `/mnt/projects/mlmi/reg2/models/` | `Qwen3-VL-8B-Instruct`, `Qwen3-VL-30B-A3B-Instruct`, `InternVL3_5-8B`, `InternVL3_5-14B`, `medgemma-1.5-4b-it` |
| `/mnt/projects/mlmi/reg2/repos/` | Cloned code: `mlmi_reg2_pathology_report_gen`, `TITAN`, `Patho-R1`, `quilt-llava` |
| `/mnt/projects/mlmi/reg2/dataset/` | Offline thumbnail caches: `thumbnails/`, `thumbnails_kmeans/`, `thumbnails_kmeans_5/` (**done**) |

---

## 2. Target architecture (hardware-limited)

**Hardware budget:** 1× A100-80G or 2× A100-40G. Models reload between phases on the same GPU(s).

```mermaid
flowchart TD
    subgraph offline [Offline preprocessing — once per WSI]
        WSI[WSI .svs]
        Tile[openslide native px @ 5×/10×/20×/40×]
        Thumb[thumbnail PNG per slide]
        CONCH[CONCHv1.5 ×4 pools → patch_embeddings_{zoom}.pt]
        TITAN20[TITAN slide encoder @ 20× only]
        WSI --> Tile --> CONCH
        WSI --> Thumb
        Tile --> TITAN20
    end

    subgraph phase1 [Phase 1 — graph traversal per WSI]
        Node[Node: zoom_level + description]
        Zoom[Load patch_embeddings_{zoom}.pt]
        Ret[CONCH text×vision cosine → raw patch images]
        HR[HippoRAG 2 top-2 CoT steps]
        VLM[Qwen2.5-VL-7B-Instruct INT8]
        CoT[cot_chain.json]
        Node --> Zoom --> Ret
        Node --> HR
        Ret --> VLM
        HR --> VLM
        VLM --> CoT
        CoT --> HR
    end

    subgraph phase2 [Phase 2 — report generation]
        Proj[slide_emb 1024 → 4096 linear]
        LLM[Qwen2.5-7B-Instruct]
        Report[CAP pathology report]
        Edges[reasoning graph edges]
        CoT --> LLM
        Proj --> LLM
        LLM --> Report
        Report --> Edges
    end

    offline --> phase1
    phase1 --> phase2
```

**Design lineage:** Phase 1/2 structure follows SlideSeek's *evidence chain → report* pattern ([arXiv:2506.20964](https://arxiv.org/abs/2506.20964)); magnification routing follows MMNavAgent's MST/CMT ideas via a **fixed uterus graph** instead of a learned navigator ([arXiv:2603.02079](https://arxiv.org/abs/2603.02079)). See §10 for borrow/skip tables.

### Phase 0 — Offline preprocessing

| Step | Spec | Artifact |
|------|------|----------|
| Thumbnail | `openslide` pyramid downsample, max edge 1024 px | `thumbnail.png` (**done** — `dataset/thumbnails/`) |
| Multi-scale tiling | Non-overlapping patches at **four zoom levels** (native px → resize to 224×224 before CONCH): | `coords_{5x,10x,20x,40x}.pt`, `meta_{zoom}.json` |
| | **5×:** 2048×2048 → 224×224 | |
| | **10×:** 1024×1024 → 224×224 | |
| | **20×:** 512×512 → 224×224 | |
| | **40×:** 256×256 → 224×224 | |
| CONCH encode | Run CONCHv1.5 **vision encoder separately at each zoom** | `patch_embeddings_{5x,10x,20x,40x}.pt` — shape `[N, 768]` each |
| TITAN slide encode | Aggregate **×20 patches only** (TITAN native training mag) | `slide_embedding.pt` (1024-d) |

**Critical split:** Four CONCH pools feed **Phase 1 retrieval only**. TITAN slide embedding uses the ×20 pool exclusively — not multi-scale aggregation.

Per-slide cache layout:

```text
cache_root/{slide_id}/
  patch_embeddings_5x.pt   coords_5x.pt
  patch_embeddings_10x.pt  coords_10x.pt
  patch_embeddings_20x.pt coords_20x.pt   ← TITAN slide_embedding.pt source
  patch_embeddings_40x.pt  coords_40x.pt
  slide_embedding.pt
```

### Phase 1 — Graph traversal

For each node in `data/graph/execution_graph.jsonl` (**deterministic** order; graph owns navigation):

**a. Zoom tier** — read `tier = node["zoom_level"]` (`5x` \| `10x` \| `20x` \| `40x`); load `patch_embeddings_{tier}.pt` for this WSI.

**b. CONCH text query** — encode node question + description (same 768-d space as vision):

```python
node_text_emb = conch_text_encoder(node.question + " " + node.description)  # (768,)
```

**c. Cross-modal retrieval** — cosine similarity within that zoom pool only; load **raw patch images** for VLM (embeddings are retrieval-only):

```python
similarities = cosine_similarity(node_text_emb, patch_embeddings[tier])
top_k = 3 if tier == "5x" else 5   # also k=3 at 10×; k=5 at 20×/40×
top_k_patches = load_raw_patches(tier, top_k_indices)
```

**d. VLM answer** — pass `top_k_patches` (raw images) + node question + HippoRAG context → Qwen2.5-VL-7B; append to `cot_chain`; online-update HippoRAG 2.

Every graph node **must** carry `zoom_level` and should include `description` for richer CONCH text queries.

Output: `runs/{slide_id}/cot_chain.json`

### Phase 2 — Report generation

1. Serialize `cot_chain` as JSON string  
2. Project `slide_emb` (1024-d) through learned linear layer → 4096-d → `[SLIDE]` prefix token  
3. Prompt Qwen2.5-7B-Instruct (text-only, same GPU after VLM unload): CoT + slide prefix → CAP-format report  
4. Parse report into reasoning-graph edges for REG² eval  

Output: `runs/{slide_id}/report.txt`, `runs/{slide_id}/pred_edges.jsonl`

---

## 3. Current codebase status (audit snapshot)

| Component | Status | Location |
|-----------|--------|----------|
| Graph loader + deterministic traversal | **Partial** — seed graph has 3 nodes; traversal works | `graph/`, `agent/controller.py` |
| Thumbnail baseline (P1) | **Done on cluster** — `dataset/thumbnails/` (+ kmeans variants) | `vision/thumbnail.py`, `scripts/vision/build_thumbnail_cache.py` |
| openslide tiling 512×512 | **Implemented** — four bands (×4/×10/×20/×40) | `vision/wsi_io.py`, `scripts/vision/tile_slides.py` |
| CONCH patch encoder (4 pools) | **Partial** — via `TitanEncoder.return_conch()`; encode all four levels offline | `vision/encoders/titan.py`, `scripts/vision/encode_patches_offline.py` |
| TITAN slide encoder | **Partial** — ×20-only slide emb; dim documented as 768 (should be 1024) | `scripts/vision/encode_slide_embeddings.py` |
| CONCH/TITAN unified offline job | **Missing** | — |
| Graph-tier zoom retrieval | **Implemented** — `node.zoom_level` → `embeddings_{band}.pt`; **primary inference path** | `retrieval/graph_guided.py`, `retrieval/titan_cosine.py` |
| Patch retrieval (cosine) | **Implemented** — TITAN **text** encoder + K-means pool | `retrieval/titan_cosine.py` |
| HippoRAG 2 | **Stub only** | `memory/hipporag2.py` |
| Per-node VLM | **Partial** — Qwen3-VL-8B via OpenAI API, not local Qwen2.5-VL-7B INT8 | `agent/backends.py` |
| Phase 2 report LLM + slide projector | **Missing** — report = last graph node answer | — |
| cot_chain / report disk persistence | **Partial** — optional `--output` on run_agent | `baselines/run_agent.py` |
| REG² chain metrics (BPV, Edge-F1, MESS) | **Implemented** | `eval/metrics/chain.py`, `eval/run_eval.py` |
| Report → edge parser | **Missing** | — |
| Deployed VLMs + sibling repos | **Cluster only** — see §1 asset table | `/mnt/projects/mlmi/reg2/models/`, `repos/` |

---

## 4. Graph artifact

| Artifact | Path | Role |
|----------|------|------|
| **Execution graph** | `data/graph/execution_graph.jsonl` | Agent walk — DOGA maintains (**schema is read-only ground truth**) |
| Ontology mirror | `data/graph/ontology_graph.jsonl` | Optional full drawio export (not present yet) |

Navigation is **deterministic**: `JsonGraphStore.next()` follows `edges` keyed by VLM answer. The LLM never chooses the next node.

Drawio labels are **medical categories**, not questions. Templated `question` fields and `interaction` types are added in JSONL.

---

## 5. JSONL node schema

| Field | Description |
|-------|-------------|
| `id`, `label`, `question`, `description` | Node identity; `description` augments CONCH text retrieval |
| `tier` | `global_features` \| `local_features` \| `integration` |
| `node_kind` | `global` \| `compartment` \| `local` \| `integration` \| `report` |
| `interaction` | `single_select` \| `multi_select` \| `boolean` \| `free_text` |
| `options`, `edges` | Answers; `__default__` for multi_select converge |
| `zoom_level` | **`5x` \| `10x` \| `20x` \| `40x`** — required on every node; selects CONCH pool |
| `visual_policy` | `thumbnail_only` \| `patch_retrieve` \| `both` |
| `root`, `is_leaf` | Traversal anchors |

---

## 6. Memory

| Layer | Target | Current |
|-------|--------|---------|
| Episodic | Partial CoT in prompt | `memory/episodic.py` ✅ |
| Semantic | HippoRAG 2 — build on train CoT, retrieve top-2, online update | `memory/hipporag2.py` stub |

Factory: `--memory flat|hipporag2|graphrag` (graphrag = ablation stub).

---

## 7. Team workflow

- Pull from `main` often; **feature branches → PR → reviewer → merge**
- Document on **ShareLaTeX** for the final report

### Owner lanes

| Person | Owns | Entry points |
|--------|------|--------------|
| **DOGA** | Graph JSONL | `data/graph/execution_graph.jsonl`, `graph/loader.py` |
| **NICK** | HippoRAG 2 | `memory/hipporag2.py`, `scripts/memory/` |
| **DOMI** | WSI offline + retrieval + pipeline | `vision/`, `scripts/vision/`, `scripts/preprocess/`, `scripts/inference/` |
| **XUN** | VLM serve (Qwen2.5-VL + Qwen2.5 text) | `configs/paths.yaml`, `agent/backends/`, `scripts/cluster/` |
| **ALL** | Eval, agent | `eval/`, `baselines/run_agent.py` |

---

## 8. Commands

```bash
# Tests
pytest tests/

# --- Target pipeline (to be implemented) ---
# Offline (cluster GPU)
python -m scripts.preprocess.run_offline_wsi --slide CASE.svs

# Phase 1 + 2 inference
python -m scripts.inference.run_phase1 --slide-id CASE.svs
python -m scripts.inference.run_phase2 --slide-id CASE.svs

# --- Current baselines (cluster: models under /mnt/projects/mlmi/reg2/models/) ---
python -m baselines.run_agent --memory flat --visual thumbnail --navigator graph_guided
python -m baselines.run_agent --backend qwen --visual patch_retrieve --retriever graph_guided --slide-id CASE.svs
# Primary path: node.zoom_level → one of four CONCH pools (not a flat pool)

# Eval
python -m eval.run_eval --pred runs/predictions.jsonl --gt data/labels/chains.jsonl --split test

# Offline vision (existing cluster jobs)
python -m scripts.vision.tile_slides --slide CASE.svs
python -m scripts.vision.encode_patches_offline --slide CASE.svs
python -m scripts.vision.encode_slide_embeddings --slide CASE.svs
```

---

## 9. Ablations (legacy + target)

**Default inference stack to try first:** `--visual patch_retrieve --retriever graph_guided` with each node’s `zoom_level` selecting its magnification-specific CONCH pool.

| Knob | Values | Notes |
|------|--------|-------|
| `--visual` | `thumbnail`, `patch_retrieve`, `slide_embed`, `none` | `thumbnail` = P1 baseline (cluster caches ready) |
| `--memory` | `flat`, `hipporag2` | |
| `--retriever` | `none`, `titan_cosine`, `graph_guided` | **`graph_guided` = default** — `zoom_level` → pool |
| `--navigator` | `graph_guided` | Same graph-as-MST policy |
| VLM | Qwen3-VL-8B API (current, cluster) → Qwen2.5-VL-7B INT8 (target) | Weights in `models/` |
| Flat ×20 pool | Ablation | Force all nodes to `embeddings_high.pt` regardless of `zoom_level` |

---

## 10. Influencing prior work — what we borrow vs skip

Two recent pathology-agent papers shape our design. We **do not** replicate either system end-to-end (hardware, data scale, and REG² eval constraints differ), but we explicitly steal specific ideas.

### PathChat+ & SlideSeek ([arXiv:2506.20964](https://arxiv.org/abs/2506.20964))

**What they do:** PathChat+ is a pathology-specific MLLM trained on ~1M instruction samples. **SlideSeek** wraps it in a multi-agent loop that iteratively inspects gigapixel WSIs through hierarchical diagnostic reasoning and produces visually grounded summary reports (strong on DDxBench).

| Idea from PathChat+ / SlideSeek | Our adoption |
|-----------------------------------|--------------|
| **Evidence-based per-step reasoning** — each answer grounded in selected patch images, not slide-level guess | ✅ Phase 1: top-k CONCH-retrieved patches → Qwen2.5-VL per graph node |
| **Iterative chain accumulation** — partial reasoning carried forward across steps | ✅ `cot_chain` list + episodic memory + HippoRAG 2 retrieval of similar past steps |
| **Hierarchical diagnostic structure** — coarse → fine reasoning over a case | ✅ Uterus ontology tiers (`global_features` → `local_features` → `integration`); deterministic graph replaces SlideSeek's learned agent planner |
| **Separate synthesis stage** — chain of evidence → final human-readable report | ✅ Phase 2: dedicated Qwen2.5-7B report writer (not the same call as node answering) |
| **Visually grounded outputs** — answers tied to WSI regions | ✅ Retrieved patch PNGs in VLM prompt; optional edge parser links answers to graph nodes |
| **PathChat+ as the VLM** | ❌ Too large for 1–2× A100-40G; we use **Qwen2.5-VL-7B-Instruct INT8** instead |
| **Autonomous multi-agent navigation** (SlideSeek agents pick where to look next) | ❌ REG² requires reproducible reasoning paths → **deterministic graph traversal** owns navigation |
| **End-to-end SlideSeek training** | ❌ Out of scope; we reuse MahmoodLab encoders + open VLMs |

**Net effect:** SlideSeek validates the *shape* of our pipeline (retrieve evidence → answer step-by-step → synthesize report). We swap their proprietary agent stack for a **fixed uterus graph + CONCH retrieval + HippoRAG memory + smaller open VLMs**.

### MMNavAgent ([arXiv:2603.02079](https://arxiv.org/abs/2603.02079))

**What they do:** Two tools in a closed loop — **MST** (magnification selection agent) and **CMT** (cross-magnification navigation with attention heatmaps). An LLM starts from a thumbnail, picks zoom/actions, and aggregates features across adjacent magnifications.

| Idea from MMNavAgent | Our adoption |
|----------------------|--------------|
| **Thumbnail as global anchor** before patch-level inspection | ✅ P1 baseline + optional `visual_policy: both` (thumbnail + retrieved patches) |
| **Magnification should match question granularity** | ✅ **Primary path:** each node’s `zoom_level` (`4x`/`10x`/`20x`/`40x`) selects a separate pre-extracted CONCH pool via `GraphGuidedRetriever` — **graph-as-MST** |
| **Cross-magnification context (CMT)** — fuse adjacent-scale features for the same region | ✅ **Shipped with graph-tier retrieval:** when retrieving at ×20, also load ×10 **parent patch** (`retrieval/titan_cosine.py`, `find_parent_patch_index`) |
| **Navigation memory bank** of prior zoom/move steps | ✅ Partial analogue: episodic CoT + HippoRAG 2 (text memory, not spatial action log) |
| **Trained MST policy** (slide-label supervision) | ❌ We have explicit graph tiers instead of learning zoom policy |
| **Full MST ↔ CMT agent loop** | ❌ Too complex for 220-slide course project; hook stub in `vision/navigation.py` only |
| **Attention heatmap region proposal** | ❌ Not implemented; CONCH cosine sim replaces CMT heatmaps for patch selection |

**Net effect:** MMNavAgent tells us *how* to think about magnification — we encode that knowledge in the **ontology graph**, and steal only the **CMT adjacent-scale fusion** idea as an optional retrieval enricher.

---

## 11. References

| Resource | Link |
|----------|------|
| PathChat+ & SlideSeek | [arXiv:2506.20964](https://arxiv.org/abs/2506.20964) |
| MMNavAgent | [arXiv:2603.02079](https://arxiv.org/abs/2603.02079) |
| TITAN / CONCH | MahmoodLab encoders |
| HippoRAG 2 | [github.com/OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) |
| Qwen2.5-VL / Qwen2.5 | Node VLM + report LLM |
| Patho-R1, MedMemoryBench | CoT supervision & memory benchmarks |
