# Training (DOMI) — Phase-1 LoRA node answerer

LoRA fine-tune the **Phase-1 node answerer** (default **Qwen3-VL-8B-Instruct**) so it
answers each graph node from the retrieved patches better than zero-shot.

Pipeline (mirrors inference for train/serve parity):

1. WP3 chains → `data/labels/chains.jsonl` (`scripts/cluster/build_chains.sh`)
2. `training/dataset.py::build_training_jsonl` — replay the **same** `--visual patch_retrieve`
   + `graph_guided` pathway per GT node → `training/samples.jsonl`
3. `training/lora.py::train_lora` — multimodal LoRA SFT (transformers + peft + trl)
4. Serve: merge adapter → vLLM (`ZeroShotQwenBackend`) **or** `FineTunedBackend` (HF)
5. Evaluate Edge-F1 vs zero-shot on the test split

## What one training sample looks like

For each **train** slide and each **single_select / boolean** node on its GT path in
`chains.jsonl`:

- `system` = `agent/prompts.py::STEP_A_SYSTEM`
- `user`   = `format_step_a_user(node, prior_steps)` (question + guidance + allowed keys + prior answers)
- `images` = whole-slide thumbnail (+ CONCH top-k patches for `patch_retrieve` nodes),
  saved per node under `cache_dir/train_samples/<node_id>/`
- `target` = GT answer as JSON `{"answer_key": <gt>, "rationale": "", "confidence": 1.0}`
  (matches `agent/backends.complete_json`, i.e. `--structured-answer` / `--node-react`)

One `ChainSample` per (slide, node) — **no cross-slide pooling** (v1). The trainer masks
everything except the assistant answer tokens (completion-only loss).

> Limitation (v1): `rationale` is empty and `confidence` is always `1.0`; we only supervise
> `answer_key`. Fine for Edge-F1; revisit if you want calibrated confidence / rationales.

## Run it (cluster)

Two **separate** jobs — the data build uses the TITAN pin (`transformers==4.46`), training
needs `transformers>=4.57` for Qwen3-VL. Do not mix the envs.

```bash
# 0) prerequisites: chains.jsonl + offline 20x caches + thumbnails for train slides
sbatch scripts/cluster/build_chains.sh          # if not done

# 1) build training/samples.jsonl (GPU: CONCH text-encode + WSI crops)
sbatch scripts/cluster/build_training_jsonl.sh  # env: SPLIT=train LIMIT=0 OUTPUT=...

# 2) LoRA fine-tune (1x 80G A100)
sbatch scripts/cluster/train_lora.sh            # env: EPOCHS LR LORA_R BATCH_SIZE GRAD_ACCUM LOAD_IN_4BIT=1
```

Local module entry points (same as the sbatch wrappers):

```bash
python -m scripts.training.build_training_jsonl --output training/samples.jsonl --split train
python -m scripts.training.run_lora --train-jsonl training/samples.jsonl \
    --output-dir "$WORK/lora/qwen3vl-uterus/adapter" --epochs 3 --lr 1e-4 --lora-r 16
# low VRAM: add --load-in-4bit (QLoRA)
```

## Serve the fine-tuned model

Paths live in `configs/paths.yaml::finetuned` (`base_model`, `adapter_dir`, `merged_dir`).

**Option A — merge + vLLM (recommended; reuses the existing OpenAI path):**

```bash
python -m scripts.training.merge_lora \
    --adapter-dir "$WORK/lora/qwen3vl-uterus/adapter" \
    --output-dir  "$WORK/lora/qwen3vl-uterus/merged"
# then point qwen.model_path/vllm serve at merged_dir and run the normal --backend qwen path
```

**Option B — HF `FineTunedBackend` (no server):**

```bash
python -m baselines.run_agent --backend finetuned --visual patch_retrieve \
    --retriever graph_guided --slide-id YOUR.svs   # add --structured-answer in batch runs
```

`FineTunedBackend` (`agent/backends.py`) loads `base_model` + `adapter_dir` and implements
`answer` + `complete_json`, so `--structured-answer` and `--node-react` work unchanged.

## Evaluate

```bash
# run the fine-tuned backend over the test split, then:
python -m eval.run_eval --pred runs/predictions.jsonl --gt data/labels/chains.jsonl --split test
```

## Future ablation — KEEP-style ontology grouping (not implemented)

**Idea (from KEEP / ontology-aware pathology VLM training):** pool image–text pairs by
**graph node** (and optionally by `answer` / ontology branch) across many train WSIs, so
each reasoning step sees multiple visually consistent exemplars for the same diagnostic
concept — not just one slide's top-k patches.

**Why it can help:** rare nodes (e.g. `serous`, `carcinosarcoma`) get more visual diversity;
the model learns node-specific visual anchors tied to the uterus ontology hierarchy
(`tier`, `zoom_level`).

**Why we defer it:**

- Graph paths are slide-specific — most nodes appear on only a subset of cases; naive pooling
  mixes different compartments/contexts under the same node id
- Inference uses per-slide retrieval only — heavy cross-slide grouping can cause train/serve
  skew unless sampling is careful (e.g. 1 primary slide patch + K ontology peers)

**Planned hook (v2 ablation):** `build_training_jsonl(..., group_by_node=True)` → index patches
by `(node_id, answer)` across train slides; optional cap per group. Compare LoRA val Edge-F1 vs
v1 per-slide samples.

See Patho-R1 for pathology CoT SFT practices.
