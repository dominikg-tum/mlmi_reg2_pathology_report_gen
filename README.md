# MLMI REG² — Interactive Pathology Report Generation

TUM MLMI Practical Course · Summer 2026 · Dr. Han Li

Step-wise diagnostic reasoning from uterine WSIs → **reasoning chain** + **final pathology report**.

## Documentation


| Doc                                                      | Purpose                                 |
| -------------------------------------------------------- | --------------------------------------- |
| **[docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** | Architecture, WPs, team lanes, commands |
| [docs/cluster_setup.md](docs/cluster_setup.md)           | Garching cluster, enroot, SLURM, vLLM   |


## Team workflow

Pull from `main` · feature branches · PR + reviewer · ShareLaTeX final report.

Entry points for first TODOS per person:


| Person | Lane           | Entry points                                             |
| ------ | -------------- | -------------------------------------------------------- |
| DOGA   | Graph JSONL    | `data/graph/execution_graph.jsonl`                       |
| NICK   | Semantic RAG   | `memory/hipporag2.py`, `memory/graphrag.py`              |
| DOMI   | WSI, WP3, LoRA | `vision/`, `scripts/vision/`, `extraction/`, `training/` |


**DOMI quick commands (cluster):**

```bash
# Baseline 1 — blurry thumbnail (no GPU)
python -m scripts.vision.build_thumbnail_cache --slide YOUR.svs
python -m baselines.run_agent --backend qwen --visual thumbnail --slide-id YOUR.svs

# Baseline 2 — TITAN slide embedding (GPU + HF token)
python -m scripts.vision.encode_slide_embeddings --slide YOUR.svs
python -m baselines.run_agent --backend qwen --visual slide_embed --slide-id YOUR.svs

# P2 — patch retrieval (tile → verify → encode → k-means → demo)
# See retrieval/README.md and vision/README.md for full cluster pipeline.
```

See `vision/README.md` for sbatch jobs.

| XUN    | VLM serve      | `configs/paths.yaml`, `scripts/cluster/`                 |

## Repository structure

```text
graph/           JSONL loader + schema
agent/           controller, memory API, VLM backends
memory/          episodic + semantic RAG stubs
vision/          thumbnail (P1), offline encode, navigation hook
retrieval/       pluggable PatchRetriever (titan_cosine)
eval/            chain + report metrics
baselines/       run_agent.py
extraction/      qa_extractor.py
training/        LoRA stubs
scripts/         manifest, vision cache jobs
data/graph/      execution_graph.jsonl (seed)
```

## Dev environment

Python **3.11**. Use [uv](https://docs.astral.sh/uv/) — `.venv` is gitignored.

```bash
cd mlmi_reg2_pathology_report_gen
uv venv                    # creates .venv/
source .venv/bin/activate  # optional; uv run works without it
uv sync --extra dev        # install from uv.lock
uv run pytest tests/
```

Cluster container (openslide + torch + transformers):

```bash
uv pip install -e ".[dev,cluster]"
```

Fallback without uv: `pip install -r requirements.txt`

```bash
pytest tests/

# P1 agent — thumbnail, no TITAN
python -m baselines.run_agent --backend dummy --memory flat --visual thumbnail

# With Qwen (cluster)
python -m baselines.run_agent --backend qwen --visual thumbnail --slide-id YOUR.svs

# Eval
python -m eval.run_eval --pred runs/pred.jsonl --gt data/labels/chains.jsonl

# Manifest
python scripts/data/build_manifest.py --example-only
```

## Cluster

`/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen` — see [cluster_setup.md](docs/cluster_setup.md) (enroot quick-start, SLURM, vLLM).

Shared assets: `containers/` (team `.sqsh` bases + per-person exports), `models/` (Qwen3-VL, InternVL, MedGemma). **Create your own container** per the §3 tutorial — do not overwrite shared images. Paths: `configs/paths.yaml`. Offline WSI cache: `configs/vision.yaml` (`cache_root` → `dominik/cache` on cluster).