# Training (DOGA) — Phase-1 LoRA node answerer

LoRA fine-tune the **Phase-1 node answerer** (default **Qwen3-VL-8B-Instruct**) so it
answers each graph node from the retrieved patches better than zero-shot.

Pipeline (mirrors inference for train/serve parity):

1. WP3 chains → `data/labels/chains.jsonl` (`scripts/cluster/build_chains.sh`), with
   **`split` restamped from `data/manifests/cases.csv`**
   (`python -m scripts.data.restamp_chains_splits`). Do not train on an unrestamped file.
2. `training/dataset.py::build_training_jsonl` — replay the **same** `--visual patch_retrieve`
   + `graph_guided` pathway per GT node → `training/samples.jsonl`
3. `training/lora.py::train_lora` — multimodal LoRA SFT (transformers + peft + trl)
4. Serve: merge adapter → vLLM (`ZeroShotQwenBackend`) **or** `FineTunedBackend` (HF)
5. `scripts/training/eval_finetuned.py` — answer-key accuracy vs base on the **stamped test** split

**After a chains restamp:** rebuild `samples.jsonl` and retrain LoRA for clean `cases.csv` test
numbers. Keep any previous adapter labeled as a **legacy-split ablation** only (it may have
seen cases that `cases.csv` now marks test).

Slides whose offline embeddings are missing are skipped per-node (logged to stderr with
a summary), so a partially-encoded cache never aborts the whole build. Slide ids in
`chains.jsonl` are mapped UUID→`TUM_Uterus_XXXX` via `data/manifests/wsi_name_map.csv`
(`vision/wsi_mapping.py`) to find the cache; the on-disk `.svs` keeps its UUID.

## What one training sample looks like

For each **train** slide and each **single_select / boolean** node on its GT path in
`chains.jsonl`:

- `system` = `agent/prompts.py::STEP_A_SYSTEM`
- `user`   = `format_step_a_user(node, prior_steps)` (question + guidance + allowed keys + prior answers)
- `images` = whole-slide thumbnail (+ CONCH top-k patches for `patch_retrieve` nodes),
  saved per node under `IMAGES_OUT/<slide_id>/<node_id>/` (default `<output>/train_images`;
  the slide ICC profile is stripped so PIL can re-read the crops during training)
- `target` = GT answer as JSON `{"answer_key": <gt>, "rationale": "", "confidence": 1.0}`
  (matches `agent/backends.complete_json`, i.e. `--structured-answer` / `--node-react`)

One `ChainSample` per (slide, node) — **no cross-slide pooling** (v1). The trainer masks
everything except the assistant answer tokens (completion-only loss).

> Limitation (v1): `rationale` is empty and `confidence` is always `1.0`; we only supervise
> `answer_key`. Fine for Edge-F1; revisit if you want calibrated confidence / rationales.

## Run it (cluster)

Two **separate** jobs — the data build uses the TITAN pin (`transformers==4.46`), training
needs `transformers>=4.57` for Qwen3-VL. Do not mix the envs.

GPU partitions: `24g` works for QLoRA (`LOAD_IN_4BIT=1`) on the 8B model; `h200` has more
headroom. On `students_opportunistic` QOS a job can be preempted at launch (instant
`FAILED`, `ExitCode 0:53`) — just resubmit, or queue on `h200`.

```bash
# 0) prerequisites (once): check caches/thumbnails/WSIs for the split
python -m scripts.training.check_prereqs --split train

# 1) build training/samples.jsonl (GPU: TITAN text-encode + WSI crops; installs openslide-bin)
LIMIT=2 sbatch scripts/cluster/build_training_jsonl.sh   # smoke test first
sbatch scripts/cluster/build_training_jsonl.sh           # full: env SPLIT/LIMIT/OUTPUT/IMAGES_OUT

# 2) LoRA fine-tune
EPOCHS=2 sbatch --partition=24g scripts/cluster/train_lora.sh
#   env: EPOCHS LR LORA_R BATCH_SIZE GRAD_ACCUM LOAD_IN_4BIT=1

# 3) build the TEST split, then evaluate base vs adapter
SPLIT=test OUTPUT=$WORK/lora/test_samples.jsonl IMAGES_OUT=$WORK/lora/test_images \
    sbatch scripts/cluster/build_training_jsonl.sh
sbatch --partition=24g scripts/cluster/eval_lora.sh      # env: TEST_JSONL REPORT_DIR LIMIT
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

### Results (v1, 2026-07-24)
Test split, 314 single_select nodes, 2 epochs, r=16, QLoRA:
answer-key accuracy **0.596 (base) → 0.691 (fine-tuned), +9.6 pts**.
Biggest gains: `compartment` 0.64→0.82, `stage_extent` 0.40→0.90,
`microscopic_pattern` 0.20→0.80, `synthesis_interpretation` 0.48→0.63,
`endometrium_assessment` 0.27→0.39. Coverage limited by ~59 train slides missing
20x embeddings; rerun after they're encoded to expand the set.

`scripts/cluster/eval_lora.sh` runs `scripts/training/eval_finetuned.py` twice (base, then
adapter) over the test-split samples, printing an `OVERALL accuracy` line plus
per-interaction / per-node breakdowns, and writing `base_report.json` /
`finetuned_report.json` under `REPORT_DIR`. Accuracy = exact `answer_key` match vs the GT.

```bash
tail -60 /mnt/home/<you>/logs/lora_eval_<jobid>.out   # compare the two OVERALL accuracy lines
```

Single config (local module entry point):

```bash
python -m scripts.training.eval_finetuned \
    --test-jsonl "$WORK/lora/test_samples.jsonl" \
    --base-model "$LORA_BASE_MODEL" --adapter-dir "$LORA_ADAPTER_DIR" \
    --report "$WORK/lora/eval/finetuned_report.json"   # omit --adapter-dir for the base model
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
