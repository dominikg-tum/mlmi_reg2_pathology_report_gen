# MLMI REG² — Interactive Pathology Report Generation

TUM MLMI Practical Course · Summer 2026 · Dr. Han Li

Step-wise diagnostic reasoning from uterine WSIs → **reasoning chain** + **final pathology report**.

## Documentation


| Doc                                                      | Purpose                                 |
| -------------------------------------------------------- | --------------------------------------- |
| **[docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** | Architecture, WPs, team lanes, commands |
| [docs/cluster_setup.md](docs/cluster_setup.md)           | Garching cluster, enroot, SLURM, vLLM   |


## Team workflow

**GitHub = source of truth.** The Garching cluster = where we run heavy jobs (WSIs, GPUs, models).
Pull from `main` often · feature branches · PR + reviewer · ShareLaTeX final report.

### Where things live (don't mix these up)

| Location | What goes there | Shared? |
| -------- | --------------- | ------- |
| **GitHub** `dominikg-tum/mlmi_reg2_pathology_report_gen` | Code, configs, docs — merged via PR | Yes (whole team) |
| **`/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen`** | One team checkout for SLURM jobs | Yes — **read/pull only on cluster** |
| **`/mnt/projects/mlmi/reg2/TUMUntera/`** | WSI slides (`.svs`) | Yes — read only |
| **`/mnt/projects/mlmi/reg2/models/`** | Qwen, InternVL, MedGemma weights | Yes — read only |
| **`/mnt/projects/mlmi/reg2/containers/`** | `.sqsh` container images | Yes — **never overwrite** team images |
| **`/mnt/projects/mlmi/reg2/<yourname>/`** | Your logs, cache, embeddings | **Yours** — create `logs/`, `cache/`, `embeddings/` |

Data and models are **not** in git. Only code is. The repo on the cluster is small; slides and models live next to it under `/mnt/projects/mlmi/reg2/`.

### Day-to-day workflow (everyone)

```text
1. Laptop (IDE)             →  edit code on a feature branch
2. GitHub                   →  push branch, open PR, get review, merge to main
3. Cluster (shared clone)   →  git pull, then sbatch / srun your job
4. Personal folder          →  job logs & caches land in /mnt/projects/mlmi/reg2/<yourname>/
```

**Typical loop:**

```bash
# --- on your laptop (daily coding) ---
git checkout main && git pull
git checkout -b nick/hipporag2-stub
# ... edit, test locally if possible ...
git add . && git commit -m "NICK: stub HippoRAG2 retriever"
git push -u origin nick/hipporag2-stub
# → open PR on GitHub, ask teammate to review, merge

# --- on the cluster (before running a job) ---
ssh youruser@head
cd /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
git status          # must be clean — no uncommitted changes!
git checkout main
git pull
sbatch scripts/cluster/explore_data.sh    # or your script
```

### Your options (pick what fits you)

| Option | Best for | How |
| ------ | -------- | --- |
| **A — Laptop + Cursor (recommended)** | Writing code, tests, PRs | Clone repo on your machine. Push to GitHub. Pull on cluster only to run jobs. |
| **B — IDE Remote-SSH** | Editing & debugging on real GPUs/data | See [Connect via IDE](#connect-via-ide-cursor--vs-code) below. Still push via GitHub — don't treat the shared cluster checkout as your only copy. |
| **C — Cluster terminal only** | Quick `git pull` + submit jobs | SSH to head → `git pull` in the shared clone → `sbatch`. Do **not** develop big features directly on the shared checkout. |

Most of us use **A for coding** and **pull on cluster before every job**. Use **B** when something only fails on the cluster or you need a GPU interactively.

### Connect via IDE (Cursor / VS Code)

You can edit files on the cluster from your laptop using the **Remote-SSH** extension (built into Cursor; install "Remote - SSH" in VS Code). This is **not** the same as `git` — it's a remote editor session.

**Important:** connect to a **compute node**, not the head/login node. Running Cursor/VS Code on head is forbidden and will get killed.

**Steps:**

1. SSH to head (terminal only — lightweight):
   ```bash
   ssh dominikgarstenauer@head
   ```
2. Start a Remote-SSH SLURM job from head:
   ```bash
   sbatch --partition=24g /mnt/general/examples/ssh.sh
   sleep 15 && cat ~/ssh.out
   # copy the line like: ssh -p 22445 you@essen.garching.camp.cluster
   ```
   Useful commands:
   - Clean up ssh.out:
   ```bash
   truncate -s 0 ssh.out
   ```
   - Cancel job:
   ```bash
   scancel <JOBID>
   ```
   - Check your job:
   ```bash
   squeue -u $USER
   ```
   - Check GPU status (20GB for 7-8B model, 80GB for 30-32B model):
   ```bash
   nvidia-smi
   ```
   - Run VLM (single GPU, change dtype when running on H100/H200)
   ```bash
   CUDA_VISIBLE_DEVICES=0 vllm serve \
   /mnt/projects/mlmi/reg2/models/<MODEL_NAME> \
   --host 0.0.0.0 \
   --port <PORT (take from 8000)> \
   --dtype half \
   --gpu-memory-utilization 0.95 \
   --max-model-len 8192 \
   --trust-remote-code  
   ```
   
   - Run VLM (multiple GPUs)
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve \
   /mnt/projects/mlmi/reg2/models/<MODEL_NAME> \
   --host 0.0.0.0 \
   --port <PORT (take from 8000)> \
   --dtype bfloat16 \
   --tensor-parallel-size 4 \
   --gpu-memory-utilization 0.95 \
   --max-model-len 512 \
   --max-num-seqs 1\
   --trust-remote-code  
   ```
3. On your laptop in **Cursor** (or VS Code): **Ctrl+Shift+P** → **Remote-SSH: Connect to Host** → paste that `ssh -p PORT user@node...` command.
4. Open folder: `/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen`

You now have a remote terminal + editor on a GPU node. Run `srun` / `enroot` / Python from the integrated terminal there.

**Git from the remote IDE:** Remote-SSH does not log you into GitHub automatically. Either set up a cluster SSH key ([cluster_setup.md §5](docs/cluster_setup.md)) or use SSH agent forwarding ([cluster_setup.md §6.3](docs/cluster_setup.md)). Test with `ssh -T git@github.com` in the remote terminal.

Full IDE setup (settings JSON, GitHub keys, day-to-day loop): [docs/cluster_setup.md §6](docs/cluster_setup.md).

### Rules — so we don't break each other or the cluster

**Git (shared repo checkout)**

- **Never** leave uncommitted changes in `/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen` — someone else's job may run your half-finished code.
- **Never** leave the shared clone on a feature branch overnight — keep it on `main` (or agree in chat first).
- **Never** commit directly to `main` on GitHub — always branch → PR → review → merge.
- **Always** `git pull` on the cluster before `sbatch` / `srun`.

**Cluster**

- **Never** run Cursor/VS Code Remote-SSH on the **head node** — always `sbatch` the SSH job and connect to the **compute node** (see [Connect via IDE](#connect-via-ide-cursor--vs-code)).
- **Never** run heavy work on the **head node** (no training, no big Python loops, no notebook servers on head). Use `srun` or `sbatch`.
- **Never overwrite** shared `.sqsh` files in `containers/`. Export your own: `yourname_YYYYMMDD_description.sqsh`.
- **Never** store large outputs inside the git repo. Use `/mnt/projects/mlmi/reg2/<yourname>/logs|cache|embeddings`.
- **Ask in chat** before starting a long vLLM server — someone may already have one running (`squeue`, check with XUN).

**First time on the cluster?** Full setup (SSH key, enroot, SLURM): [docs/cluster_setup.md](docs/cluster_setup.md).

### Entry points for first TODOS per person:

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

# Streamlit agent frontend
uv sync --extra frontend
streamlit run app.py

# Lightweight image baseline:
# upload one image, select a remote OpenAI-compatible VLM, run the full graph.
# No TITAN, WSI cache, or patch embeddings are loaded.

# P1 agent — thumbnail, no TITAN
python -m baselines.run_agent --backend dummy --memory flat --visual thumbnail

# With Qwen (cluster)
python -m baselines.run_agent --backend qwen --memory flat --visual thumbnail --slide-id YOUR.svs

# Batch baselines (test split; needs chains.jsonl + Qwen vLLM)
python -m scripts.inference.run_baseline_batch --baseline a --split test --dry-run
BASELINE=a sbatch --array=0-69 scripts/cluster/run_baseline_batch.sh
python -m scripts.inference.run_eval_baselines --baseline all --split test

# Eval
python -m eval.run_eval --pred runs/pred.jsonl --gt data/labels/chains.jsonl

# Manifest
python scripts/data/build_manifest.py --example-only
```

## Cluster

`/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen` — see [cluster_setup.md](docs/cluster_setup.md) (enroot quick-start, SLURM, vLLM).

Shared assets: `containers/` (team `.sqsh` bases + per-person exports), `models/` (Qwen3-VL, InternVL, MedGemma). **Create your own container** per the §3 tutorial — do not overwrite shared images. Paths: `configs/paths.yaml`. Offline WSI cache: `configs/vision.yaml` (`cache_root` → `dominik/cache` on cluster).
