# Vision (DOMI)

Two **P1 baselines** for feeding WSIs to the VLM:

| Baseline | `--visual` | Offline job | VLM input |
|----------|------------|-------------|-----------|
| **1. Blurry thumbnail** | `thumbnail` | `build_thumbnail_cache.py` | Native pyramid downsample PNG |
| **2. TITAN slide embed** | `slide_embed` | `encode_slide_embeddings.py` | Thumbnail + 3 evidence patches (+ cached 768-d vector) |

Cache root: `configs/vision.yaml` → `cache_root` (cluster: `/mnt/projects/mlmi/reg2/dominik/cache`).

Per slide:

```text
{cache_root}/{slide_id}/
  thumbnail.png           # baseline 1 (+ also written by baseline 2)
  slide_embedding.pt      # baseline 2 — TITAN [768]
  evidence/patch_*.png    # baseline 2 — tissue patches for Qwen
  embeddings_high.pt      # optional — for P2 retrieval
  meta.json
```

## Local / cluster — baseline 1 (thumbnail)

```bash
cd /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen

# One slide smoke test
python -m scripts.vision.build_thumbnail_cache --slide YOUR.svs

# All slides
python -m scripts.vision.build_thumbnail_cache

# Agent
python -m baselines.run_agent --backend qwen --visual thumbnail --slide-id YOUR.svs
```

## Cluster sbatch — baseline 1

```bash
cd /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
chmod +x scripts/cluster/build_thumbnail_cache.sh

# Smoke: one slide
SLIDE=YOUR.svs sbatch scripts/cluster/build_thumbnail_cache.sh

# Batch (first 10)
LIMIT=10 sbatch scripts/cluster/build_thumbnail_cache.sh
```

## Cluster — baseline 2 (TITAN slide embedding)

**Prerequisites**

1. HuggingFace account + access to [MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN)
2. `export HF_TOKEN=hf_...` in the sbatch shell (or `huggingface-cli login` inside container)
3. GPU job (CONCH + TITAN load on CUDA)

```bash
# Interactive smoke (one slide)
srun --partition=24g --qos=students_normal --gres=gpu:1 --mem=32G --pty bash -l
enroot start --rw --mount /mnt:/mnt dominik_mlmi
cd /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
pip install openslide-python pillow transformers torch huggingface_hub
export HF_TOKEN=hf_...
python -m scripts.vision.encode_slide_embeddings --slide YOUR.svs --max-patches 256
```

```bash
# Overnight batch
chmod +x scripts/cluster/encode_titan_slides.sh
SLIDE=YOUR.svs sbatch scripts/cluster/encode_titan_slides.sh
# or all slides:
sbatch scripts/cluster/encode_titan_slides.sh
```

```bash
# Agent with slide_embed visual mode
python -m baselines.run_agent --backend qwen --visual slide_embed --slide-id YOUR.svs
```

## P2 — patch retrieval (later)

```bash
python -m scripts.vision.encode_patches_offline.py --level high
python -m baselines.run_agent --visual patch_retrieve --retriever titan_cosine
```

Data WSIs: `/mnt/projects/mlmi/reg2/TUMUntera` (see `configs/paths.yaml`).
