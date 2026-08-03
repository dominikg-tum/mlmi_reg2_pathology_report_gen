# Fine-tuning process (LoRA v1) — Qwen3-VL-8B uterus node answerer

Owner: **Doga** · Scope: Phase-1 graph node answering (not report text SFT) · Status: **complete**

Runnable commands and
code entry points live in [`training/README.md`](../../training/README.md). Numbers and file
paths for slides live under [`artifacts/lora_v1/`](../../artifacts/lora_v1/) and
[`RESULTS_PHASE1.md`](RESULTS_PHASE1.md).

---

## 1. Goal

Improve the Phase-1 **node answerer** so that, while walking the uterus diagnostic graph, the
VLM picks the correct structured `answer_key` more often than zero-shot Qwen3-VL-8B-Instruct.

We evaluate two complementary questions:

1. **Node-level (prebuilt samples):** given GT path context + the same images used at train
   time, does LoRA raise exact `answer_key` accuracy?
2. **Full agent (ablation):** end-to-end graph walk + report on the held-out **test** split,
   for two visual setups × base vs LoRA.

---

## 2. What was fine-tuned (and what was not)

| Included | Excluded |
|----------|----------|
| Multimodal LoRA on Qwen3-VL-8B-Instruct | Full-model / full-vision fine-tune |
| Supervision on GT `answer_key` JSON | Free-text report generation SFT |
| Vision encoder **frozen** | Training on unrestamped / leaking splits |
| Visual path ≈ baseline **p0** (thumbnail + CONCH patches via `graph_guided`) | HippoRAG / RAG memory |

Training sample construction replays the inference visual pathway
(`patch_retrieve` + `graph_guided`) so train and serve stay aligned. See
`training/dataset.py` and `training/README.md`.

---

## 3. Data

1. Ground-truth chains: `data/labels/chains.jsonl` with `split` restamped from
   `data/manifests/cases.csv` (151 train / 70 test in the canonical CSV; **68** test cases
   were runnable with `extraction_status=ok` for the agent ablation).
2. Build train JSONL: `scripts/training/build_training_jsonl.py` → multimodal samples
   (system + user + images + target JSON).
3. Build test JSONL the same way for node-level eval.
4. Slides without offline 20× embeddings are skipped per-node (logged); this limits coverage
   but does not abort the build.

**Important:** do not train on an unrestamped `chains.jsonl`. After any restamp, rebuild
samples and treat older adapters as legacy.

---

## 4. Hyperparameters (v1 adapter)

| Setting | Value |
|---------|------:|
| Base model | Qwen3-VL-8B-Instruct |
| Method | LoRA / QLoRA (`load_in_4bit`) |
| Rank \(r\) | 16 |
| α | 32 |
| Dropout | 0.05 |
| Target modules | q/k/v/o + gate/up/down_proj |
| Learning rate | 1e-4 |
| Schedule | cosine, warmup ratio 0.03 |
| Batch × accum | 1 × 8 |
| Precision | bf16 |
| Epochs | **2** |
| Logged steps | 160 global steps, 32 loss points |
| Vision | frozen |

Cluster launch: `scripts/cluster/train_lora.sh` → `scripts/training/run_lora.py`.
Canonical weights: `/mnt/home/dogakonuk/lora/qwen3vl-uterus/adapter/`  
Shared inference copy: `/mnt/projects/mlmi/reg2/dogakonuk/lora/qwen3vl-uterus/adapter/`

---

## 5. Training dynamics & overfitting story

- Train loss falls from ~**1.49 → 0.045** (min ~**0.017**). Curve:
  `artifacts/lora_v1/training/train_loss.png` (+ CSV / summary JSON).
- **No validation split** and no `eval_loss` were logged in v1.
- Overfit check = **held-out test** metrics, not a val curve:
  - Node-level: base **0.596** → LoRA **0.691** (+9.6 pts).
  - Full agent: large gains on CoT metrics (Edge-F1 / node acc / BPV); report ROUGE stays
    low and does not improve (expected: no report SFT).

Phase 2 (train a second adapter with val) was **not** pursued for the final report; v1 is
the sole adapter referenced below.

---

## 6. Serving LoRA at inference

Two practical paths were used:

| Path | When | How |
|------|------|-----|
| **HF `FineTunedBackend`** | Baseline **a** (thumbnail-only; no TITAN in the same env) | `BACKEND=finetuned`, `MLMI_ADAPTER_DIR=...` |
| **vLLM + `--enable-lora`** | Baseline **p0** (needs TITAN pin `transformers==4.46`, which conflicts with LoRA’s `>=4.57`) | Serve base Qwen with LoRA module name `uterus_lora`; clients use `BACKEND=qwen` and `model_name=uterus_lora` |

Full weight merge was attempted but **OOM** on available GPUs/CPU save; abandoned in favor of
vLLM LoRA serve (`--max-model-len 4096`, high GPU util).

PNG ICC profiles from WSI/patch crops can blow vLLM’s PIL `MAX_TEXT_CHUNK`; inference
re-encodes images as JPEG in `vision/vlm_messages.py`.

---

## 7. Ablation design 

Locked definitions:

| Code | Visual | Memory | Backend variants |
|------|--------|--------|------------------|
| **a** | Graph walk + **thumbnail only** | flat (episodic), no RAG | base Qwen vs LoRA |
| **p0** | Graph walk + thumbnail + **CONCH patches** (`patch_retrieve` + `graph_guided`) | flat, no RAG | base Qwen vs LoRA |

LoRA training visuals match the **p0** pathway; evaluating **a×LoRA** still tests whether
the adapter helps when patches are withheld at test time.

Full test set → CoT + report → `eval.run_eval` (rescored with report node stripped from
path metrics; Final Diag Acc = exact `diagnosis` node key match). Scripts:

- `scripts/training/rescore_ablation_arms.py`
- `scripts/training/plot_ablation_metrics.py` → **separate** CoT and Report PNGs

---

## 8. Metrics the team asked for

### CoT (reasoning path)

| Metric | Definition / note |
|--------|-------------------|
| Binary Path Validity (BPV) | Exact node-path match; **excludes** trailing `report` node |
| Edge-F1 | (node_id, normalized answer) set F1; report node excluded |
| Node accuracy | Per-GT-node answer match rate |
| MESS | Answer-text similarity (ST if available; else token overlap) |
| Final diagnosis accuracy | Exact match of `diagnosis` node `answer_key` vs GT (graph keys; not Xun 6-way map) |

### Report

| Metric | Note |
|--------|------|
| ROUGE-L, BLEU-4, clinical token-F1 | Standard report metrics |
| BERTScore | Optional; skipped in light rescore |
| Numeric fidelity / negation | Available in `eval.run_eval` |

Artifacts: `artifacts/lora_v1/ablation/` (metrics, plots, tables).

---

## 9. Reproduction checklist (cluster)

```bash
# Code pin (personal)
cd /mnt/projects/mlmi/reg2/dogakonuk/repos/mlmi_reg2_pathology_report_gen
git checkout doga/eval && git pull

# Node-level eval reports already under home; rebuild only if needed:
#   sbatch scripts/cluster/eval_lora.sh

# Rescore existing full-agent predictions (no regeneration):
python3 -m scripts.training.rescore_ablation_arms \
  --runs-root /mnt/projects/mlmi/reg2/dogakonuk/runs \
  --gt data/labels/chains.jsonl \
  --split test --skip-bert --plot

# Pull evidence into the repo tree (from laptop):
CLUSTER_SSH_HOST=dogakonuk@head.garching.camp.cluster \
  bash scripts/local/fetch_lora_v1_artifacts.sh
```

---

## 10. Related code map

| Piece | Location |
|-------|----------|
| LoRA train | `training/lora.py`, `scripts/training/run_lora.py` |
| Sample build | `training/dataset.py`, `scripts/training/build_training_jsonl.py` |
| Node eval | `scripts/training/eval_finetuned.py` |
| Loss plot | `scripts/training/plot_lora_training.py` |
| Ablation batch | `scripts/inference/run_baseline_batch.py`, `scripts/cluster/run_baseline_batch.sh` |
| Metrics | `eval/metrics/chain.py`, `eval/run_eval.py` |
| Image encode fix | `vision/vlm_messages.py` |

---
