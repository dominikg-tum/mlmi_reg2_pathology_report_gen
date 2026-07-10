# Training (DOMI)

1. WP3 chains → `data/labels/chains.jsonl`
2. `training/dataset.py` — build samples with **same** `--visual` / `--retriever` as inference
3. `training/prompt.py` — turn each `ChainSample` into Qwen3-VL chat messages (train/serve parity)
4. `training/lora.py` — LoRA run on cluster (Qwen3-VL-8B)
5. `FineTunedBackend` in `agent/backends.py` (`backend=lora`) loads the adapter for inference

See Patho-R1 for pathology CoT SFT practices.

## Run order (cluster)

```bash
# 1. Build the training data (TITAN + patch cache; transformers==4.46.0 container)
sbatch scripts/cluster/build_lora_dataset.sh          # -> $USER_ROOT/training/samples_train.jsonl (+ images/)

# 2. LoRA train (RECENT transformers; SEPARATE container/session from step 1)
sbatch scripts/cluster/train_lora.sh                  # -> $USER_ROOT/training/lora/qwen3vl_8b_v1

# 3. Inference with the fine-tuned adapter
LORA_ADAPTER_DIR=$USER_ROOT/training/lora/qwen3vl_8b_v1 \
  python -m scripts.inference.run_test_baseline_batch --backend lora --split test ...
```

**Parity:** `training/prompt.py` reuses `agent.backends.build_answer_prompt`,
`system_prompt_for`, and `visual_note_for_paths`, so the fine-tuned model trains on
byte-identical prompts to what `FineTunedBackend` sends at serve time.

**Container split:** the data builder pins `transformers==4.46.0` (TITAN requirement); the
LoRA trainer needs `transformers>=4.57` for Qwen3-VL. Keep them in separate enroot
sessions/containers.

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
