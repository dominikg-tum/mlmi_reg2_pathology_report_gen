# Training (DOMI)

1. WP3 chains → `data/labels/chains.jsonl`
2. `training/dataset.py` — build samples with **same** `--visual` / `--retriever` as inference
3. `training/lora.py` — LoRA run on cluster
4. Register `FineTunedBackend` in `agent/backends.py`

See Patho-R1 for pathology CoT SFT practices.

## Default LoRA data (v1 — per-slide, matches inference)

For each train slide and each node on its GT path in `chains.jsonl`:

- Load visuals the same way Phase 1 does (`thumbnail` or `patch_retrieve` + `graph_guided` retriever at `node.zoom_level`)
- Pair with `question`, `target_answer` (GT edge key), and episodic context from prior chain steps
- One `ChainSample` per (slide, node) — **no cross-slide pooling**

This keeps train/test distribution aligned: at inference the VLM sees patches retrieved from **that slide only**.

## Future ablation — KEEP-style ontology grouping (not implemented)

**Idea (from KEEP / ontology-aware pathology VLM training):** pool image–text pairs by **graph node** (and optionally by `answer` / ontology branch) across many train WSIs, so each reasoning step sees multiple visually consistent exemplars for the same diagnostic concept — not just one slide’s top-k patches.

**Why it can help:** rare nodes (e.g. `serous`, `carcinosarcoma`) get more visual diversity; the model learns node-specific visual anchors tied to the uterus ontology hierarchy (`tier`, `zoom_level`).

**Why we defer it:**

- Graph paths are slide-specific — most nodes appear on only a subset of cases; naive pooling mixes different compartments/contexts under the same node id
- Inference uses per-slide retrieval only — heavy cross-slide grouping can cause train/serve skew unless sampling is careful (e.g. 1 primary slide patch + K ontology peers)
- Blockers first: `chains.jsonl`, offline CONCH caches, and a working per-slide `build_training_jsonl` baseline

**Planned hook (v2 ablation):** `build_training_jsonl(..., group_by_node=True)` → index patches by `(node_id, answer)` across train slides; optional cap per group. Compare LoRA val Edge-F1 vs v1 per-slide samples.
