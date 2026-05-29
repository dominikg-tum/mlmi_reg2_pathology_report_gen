# MLMI REG² — Interactive Pathology Report Generation

TUM MLMI Practical Course · Summer 2026 · Dr. Han Li  
Build a step-wise diagnostic reasoning system from uterine WSI inputs → reasoning chain + final pathology report.

---

## Documentation

| Doc | Description |
|---|---|
| [docs/project_overview.md](docs/project_overview.md) | Goals, work packages, architecture, build order |
| [docs/cluster_setup.md](docs/cluster_setup.md) | Garching cluster, enroot, SLURM, Cursor, Qwen |

---

## Cluster Quick Start

Project root on the Garching cluster:

```bash
ls /mnt/projects/mlmi/reg2
```

| Path | Contents |
|---|---|
| `TUMUntera/` | WSI dataset (`.svs`) |
| `containers/` | Team `.sqsh` images |
| `repos/` | Git repos — **clone this repo here** |
| `models/` | Local VLMs (Qwen) |
| `dominik/` | Personal workspace + logs |

**Dominik's enroot container (already set up):**

```bash
# Created once from qwen25_dev_updated.sqsh
enroot create --name dominik_mlmi \
  /mnt/projects/mlmi/reg2/containers/qwen25_dev_updated.sqsh

# On a compute node (after srun):
enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp dominik_mlmi
```

Full setup: [docs/cluster_setup.md](docs/cluster_setup.md)

---

## Repository Structure

```text
mlmi_reg2_pathology_report_gen/
├── README.md
├── requirements.txt              # repo-specific Python deps
├── requirements_clean.txt        # mirror of cluster base deps (reference)
├── docs/
│   ├── project_overview.md       # scientific / WP overview
│   └── cluster_setup.md          # cluster + enroot + SLURM + Cursor
├── configs/
│   └── paths.yaml                # cluster path constants
├── graph/                        # ★ the heart of the project
│   ├── diagnostic_graph.py       # hard-coded graph (Han's tree) as data
│   └── controller.py             # deterministic traversal (code owns navigation)
├── extraction/
│   └── qa_extractor.py           # WP3: Q→A extraction from reports (local Qwen)
├── retrieval/
│   └── titan_retriever.py        # TITAN context-aware cosine retrieval (G1+G2)
├── baselines/
│   └── zero_shot.py              # WP4: full loop with an untrained Qwen backend
├── notebooks/
│   └── explore_wsi.ipynb         # WP1 exploration
├── scripts/
│   └── cluster/                  # SLURM batch scripts
└── tests/
    └── test_graph.py             # graph integrity + traversal smoke test
```

**Why graph-centric?** The graph + controller own traversal deterministically; the
model only answers one node at a time and never decides where to go. Fine-tuning
swaps the `AnswerBackend` behind the controller without changing navigation. See
[docs/project_overview.md](docs/project_overview.md) §5–§7.

---

## Work Packages

| WP | Task |
|---|---|
| WP1 | Explore WSIs, reports, diagnostic tree |
| WP2 | Patch extraction + TITAN encoding |
| WP3 | Q→A extraction from reports (local Qwen) |
| WP4 | Zero-shot baselines + evaluation setup |
| WP5 | G1: step-wise fine-tuning · G2: agent framework |
| WP6–WP10 | Compare, refine, integrate, benchmark, document |

See [docs/project_overview.md](docs/project_overview.md) for architecture and build order.

---

## Local Development

```bash
git clone git@github.com:YOUR_USERNAME/mlmi_reg2_pathology_report_gen.git
cd mlmi_reg2_pathology_report_gen
pip install -r requirements.txt
```

On the cluster, install inside `dominik_mlmi` — see [docs/cluster_setup.md](docs/cluster_setup.md).

---

## First Batch Job (WP1)

```bash
# On head
sbatch scripts/cluster/explore_data.sh
squeue
```

Adjust `configs/paths.yaml` and the `#SBATCH` paths in the script for your exported `.sqsh` filename.
