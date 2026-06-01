# MLMI REG² — Interactive Pathology Report Generation

TUM MLMI Practical Course · Summer 2026 · Dr. Han Li

Step-wise diagnostic reasoning from uterine WSIs → **reasoning chain** + **final pathology report**.

## Documentation

| Doc | Purpose |
|-----|---------|
| **[docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** | Architecture, WPs, team lanes, commands |
| [docs/cluster_setup.md](docs/cluster_setup.md) | Garching cluster, enroot, SLURM, vLLM |

## Team workflow

Pull from `main` · feature branches · PR + reviewer · ShareLaTeX final report.

| Person | Lane | Entry points |
|--------|------|--------------|
| DOGA | Graph JSONL | `data/graph/execution_graph.jsonl` |
| NICK | Semantic RAG | `memory/hipporag2.py`, `memory/graphrag.py` |
| DOMI | WSI, WP3, LoRA | `vision/`, `scripts/vision/`, `extraction/`, `training/` |
| XUN | VLM serve | `configs/paths.yaml`, `scripts/cluster/` |

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

## Quick start

```bash
cd mlmi_reg2_pathology_report_gen
pip install -r requirements.txt pyyaml numpy pytest

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

`/mnt/projects/mlmi/reg2` — see [cluster_setup.md](docs/cluster_setup.md).

Set `configs/vision.yaml` → `cache_root` for offline thumbnails and patch embeddings.
