# Garching Cluster Setup — MLMI REG²

Step-by-step guide for working on the TUM Garching GPU cluster (`/mnt/projects/mlmi/reg2`).

---

## 1. Project Layout on the Cluster

Your project lives at:

```bash
ls /mnt/projects/mlmi/reg2
```

| Path | Contents |
|---|---|
| `TUMUntera/` | WSI dataset (`.svs` files) |
| `containers/` | Pre-built `.sqsh` images (`qwen25_graphrag.sqsh` = GraphRAG team base) |
| `repos/` | Git repositories — **clone this project here** |
| `reg2_repo/` | Legacy folder — **do not** put `mlmi_reg2_pathology_report_gen` here |
| `models/` | Locally deployed VLMs (Qwen3-VL, InternVL, MedGemma, …) |
| `scripts/` | Shared helper scripts |
| `case_reports_to_korea_collaborators.xlsx` | Labels (`slide_ids`, `english_reports`, …) |
| `requirements.txt` / `requirements_clean.txt` | Dependencies to install inside containers |
| `dominik/` | Personal logs, cache, embeddings (`chmod 777`) |

This repository should be cloned to:

```text
/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
```

Team scripts at the project root (e.g. `extract_report_parts.py`, `report_parts_extracted.json`)
are older copies; prefer `extraction/` in the repo and outputs under `data/` (see
`configs/paths.yaml` → `extraction.report_parts_json`).

---

## 2. Golden Rules

1. **Never run heavy commands on the head node** — always start a SLURM job first (`srun` or `sbatch`).
2. **Never overwrite shared `.sqsh` files** — export your changes to a **new** file every time.
3. **Name personal containers** `yourname_YYYYMMDD_description.sqsh` so the team can track versions.
4. **Use `chmod 777`** on personal folders under `/mnt/projects/mlmi/reg2/` so teammates can read logs and outputs.

---

## 3. Container Workflow (enroot + `.sqsh`)

The cluster uses [enroot](https://github.com/NVIDIA/enroot) for containerized environments.

### 3.1 Starting from a team base image (first time)

```bash
# Import and create from the team GraphRAG base (filename on disk: qwen25_graphrag.sqsh)
enroot import -- /mnt/projects/mlmi/reg2/containers/qwen25_graphrag.sqsh
enroot create --name my_env qwen25_graphrag.sqsh
enroot start my_env
```

### 3.2 Dominik's current setup (already created)

Container **`dominik_mlmi`** was created from the Qwen 2.5 dev image:

```bash
enroot create --name dominik_mlmi \
  /mnt/projects/mlmi/reg2/containers/qwen25_dev_updated.sqsh
```

Start it on a compute node (not head):

```bash
enroot start --rw \
  --mount /mnt:/mnt \
  --mount /tmp:/tmp \
  dominik_mlmi
```

Add `--root` if you need root inside the container (e.g. for system-level installs):

```bash
enroot start --root --rw \
  --mount /mnt:/mnt \
  --mount /tmp:/tmp \
  dominik_mlmi
```

### 3.3 Install dependencies

**Local laptop** (unit tests only — no openslide/GPU):

```bash
cd mlmi_reg2_pathology_report_gen
uv venv && uv sync --extra dev
uv run pytest tests/
```

**Inside the cluster container** (GPU jobs):

```bash
cd /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen

# preferred — editable install with lockfile
uv pip install -e ".[dev,cluster]"

# or legacy pip
pip install -r /mnt/projects/mlmi/reg2/requirements_clean.txt
pip install -r requirements.txt
pip install openslide-python scikit-learn tqdm

exit
```

### 3.4 Save your container (always a new file)

```bash
enroot export --force \
  --output /mnt/projects/mlmi/reg2/containers/dominik_$(date +%Y%m%d)_base.sqsh \
  dominik_mlmi

echo "Saved! Don't forget: always save to a new file after changes."
```

---

## 4. Interactive Job & Personal Workspace

### 4.1 Start an interactive GPU session

```bash
# From head — 24G partition, one GPU (CPU/RAM auto-allocated per GPU)
srun --partition=24g --qos=students_normal --gres=gpu:1 --pty bash -l
```

### 4.2 Create your working directory

```bash
mkdir -p /mnt/projects/mlmi/reg2/dominik/{logs,cache,embeddings}
chmod 777 /mnt/projects/mlmi/reg2/dominik
```

Offline WSI caches use `dominik/cache` — see `configs/vision.yaml` (`cache_root`).

---

## 5. GitHub Setup

Run on a **compute node**, inside your container:

```bash
enroot start --rw --mount /mnt:/mnt dominik_mlmi

git config --global user.name "Dominik Garstenauer"
git config --global user.email "your@tum.de"

# SSH key for GitHub
ssh-keygen -t ed25519 -C "your@tum.de" -f ~/.ssh/id_github -N ""
cat ~/.ssh/id_github.pub
# → Add to GitHub: Settings → SSH Keys → New SSH Key

ssh -T git@github.com

cd /mnt/projects/mlmi/reg2/repos
git clone git@github.com:YOUR_USERNAME/mlmi_reg2_pathology_report_gen.git
chmod 777 mlmi_reg2_pathology_report_gen
```

Typical workflow:

```bash
cd /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
git pull
# ... make changes ...
git add .
git commit -m "WP1: explore WSI data structure"
git push
```

---

## 6. Connect Cursor to the Cluster

Cursor uses the same Remote SSH flow as VS Code.

### 6.1 Launch an SSH job from head

```bash
sbatch --partition=24g /mnt/general/examples/ssh.sh
sleep 15
cat ssh.out
# Example output:
# ssh -p 22445 dominikgarstenauer@essen.garching.camp.cluster
# code --folder-uri vscode-remote://ssh-remote+dominikgarstenauer@essen...
```

In Cursor: **Ctrl+Shift+P** → **Remote-SSH: Connect to Host** → paste the `ssh -p PORT user@node...` command, or use the `code --folder-uri` line.

Open folder: `/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen`

### 6.2 Recommended Cursor / VS Code settings

Add to **User Settings JSON** (`Ctrl+Shift+P` → *Preferences: Open User Settings JSON*):

```json
{
  "remote.SSH.remotePlatform": {
    "essen.garching.camp.cluster": "linux",
    "koblenz.garching.camp.cluster": "linux",
    "heidelberg.garching.camp.cluster": "linux",
    "muenchen.garching.camp.cluster": "linux"
  },
  "remote.SSH.serverInstallPath": {
    "essen.garching.camp.cluster": "/tmp/vscode-server",
    "koblenz.garching.camp.cluster": "/tmp/vscode-server"
  }
}
```

### 6.3 Cursor + GitHub cheat sheet

Remote-SSH and GitHub use **two separate SSH connections**:

| Connection | Target | Purpose |
|---|---|---|
| Cursor Remote-SSH | `user@essen.garching.camp.cluster:PORT` | Edit code on a compute node |
| Git over SSH | `git@github.com` | `git clone` / `pull` / `push` |

Remote-SSH does **not** authenticate you to GitHub. Git commands run in the Cursor
terminal on the cluster and need their own GitHub credentials there.

#### Option A — SSH agent forwarding (recommended)

Reuse the SSH key already on your laptop. No key duplication inside enroot.

**On your laptop** (`~/.ssh/config`):

```sshconfig
Host *.garching.camp.cluster
  ForwardAgent yes
  IdentityFile ~/.ssh/id_ed25519
```

Start the agent locally and add your key:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

After connecting via Remote-SSH, verify inside the remote terminal:

```bash
ssh -T git@github.com
# Hi YOUR_USERNAME! You've successfully authenticated...
```

Then use git normally:

```bash
cd /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
git pull
git push
```

**Pros:** one key to manage; works across container restarts.  
**Cons:** agent must be running on your laptop; forwarding may be disabled by cluster policy — test with `ssh -T git@github.com` on first connect.

#### Option B — Key inside the enroot container

Generate a dedicated key inside the container (see §5). Add the public key to
GitHub → Settings → SSH Keys.

**Pros:** works even if agent forwarding is blocked.  
**Cons:** keys live in the container filesystem — they are lost unless you export
the container to a new `.sqsh` (§3.4). Prefer storing keys in cluster home
(`~/.ssh` on the compute node, outside enroot) if your workflow allows git outside
the container.

#### Option C — HTTPS + personal access token

```bash
git remote set-url origin https://github.com/YOUR_USERNAME/mlmi_reg2_pathology_report_gen.git
git pull   # prompts for username + GitHub PAT once; cache with credential helper
```

Works everywhere but is less convenient for frequent pushes.

#### Typical day-to-day loop (Cursor + GitHub)

```bash
# 1. From head — start SSH job for Cursor
sbatch --partition=24g /mnt/general/examples/ssh.sh && sleep 15 && cat ssh.out

# 2. In Cursor: Remote-SSH → connect → open repo folder

# 3. On the remote terminal (inside enroot if needed):
enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp dominik_mlmi
cd /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
git pull

# 4. Edit in Cursor, then commit from the remote terminal:
git add .
git commit -m "WP3: extend qa_extractor"
git push
```

Keep the repo on `/mnt/projects/mlmi/reg2/repos/` — not in `/tmp` or a
container-only path — so it persists across sessions.

---

## 7. Running Models on the Cluster

All model paths and API endpoints are centralized in
[`configs/paths.yaml`](../configs/paths.yaml). Update that file once you confirm
the live model name / port with teammates.

### 7.1 What's deployed where

```bash
ls /mnt/projects/mlmi/reg2/containers/    # team + personal .sqsh exports (§3.0)
ls /mnt/projects/mlmi/reg2/models/        # local model weights (§3.0)
ls /mnt/projects/mlmi/reg2/repos/         # TITAN, Patho-R1, quilt-llava, …
grep -i qwen /mnt/projects/mlmi/reg2/scripts/*.sh   # team helper scripts
```

| Asset | Path | Role |
|---|---|---|
| **Qwen3-VL-8B-Instruct** | `models/Qwen3-VL-8B-Instruct` | MVP VLM — default in `configs/paths.yaml` and `start_qwen_server.sh` |
| **Qwen3-VL-30B-A3B-Instruct** | `models/Qwen3-VL-30B-A3B-Instruct` | Larger upper-bound baseline |
| **InternVL3_5-8B** | `models/InternVL3_5-8B` | LoRA fine-tune candidate |
| **InternVL3_5-14B** | `models/InternVL3_5-14B` | LoRA fine-tune candidate |
| **medgemma-1.5-4b-it** | `models/medgemma-1.5-4b-it` | Small medical VLM baseline |
| **TITAN** | `repos/TITAN` | Frozen image + text encoders for retrieval (WP2) |
| **Patho-R1** | `repos/Patho-R1` | Pathology reasoning baseline |

Full on-disk inventory: [§3.0](#30-shared-containers--models-inventory). Paths in [`configs/paths.yaml`](../configs/paths.yaml) under `models:`.

Repo scripts live under `scripts/cluster/`:

```text
scripts/cluster/
├── load_paths.sh           # source of truth reader for paths.yaml
├── start_qwen_server.sh    # vLLM OpenAI-compatible API (long-running)
└── explore_data.sh         # WP1 notebook batch job (one-shot)
```

SLURM scripts `source load_paths.sh` so container and model paths stay in sync with
`configs/paths.yaml`. When adding jobs, copy one of these as a template — same `#SBATCH`
headers, `load_cluster_paths`, `enroot start --root --rw --mount /mnt:/mnt`, and logs under
`dominik/logs/`.

### 7.2 Qwen via vLLM (generative VLM / text)

The cluster has **no public LLM API**. Models are served locally with
[vLLM](https://docs.vllm.ai/) inside enroot.

#### Check if a server is already running

Ask teammates first — a shared vLLM job may already be up. Otherwise:

```bash
squeue -u $USER
# If you started one yourself:
tail -f /mnt/projects/mlmi/reg2/dominik/logs/qwen_server_*.out
curl -s http://localhost:8000/v1/models | python -m json.tool
```

#### Start your own vLLM server (batch)

From the repo root (on head — `sbatch` is lightweight):

```bash
cd /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen

# Model + container come from configs/paths.yaml (qwen.*, user.container_sqsh)
sbatch scripts/cluster/start_qwen_server.sh

squeue
tail -f /mnt/projects/mlmi/reg2/dominik/logs/qwen_server_<job-id>.out
# Wait for: "Uvicorn running on http://0.0.0.0:8000"
```

The script requests 2 GPUs and 60G RAM (tune in the script for 30B). To switch models,
edit `qwen.model_path` and `qwen.model_name` in `configs/paths.yaml` (e.g. to
`models.qwen3_vl_30b`), then re-submit.

#### Start interactively (debugging)

```bash
srun --partition=24g --qos=students_normal --gres=gpu:2 --pty bash -l

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  /mnt/projects/mlmi/reg2/containers/qwen25_dev_updated.sqsh \
  python -m vllm.entrypoints.openai.api_server \
    --model /mnt/projects/mlmi/reg2/models/Qwen3-VL-8B-Instruct \
    --port 8000
```

Keep this terminal open — the server runs in the foreground.

#### Call the server from repo code

Point `configs/paths.yaml` at the running endpoint, then:

```bash
# Smoke test (reads api_base_url + model_name from paths.yaml)
python extraction/qa_extractor.py

# Full report-parts extraction (batch over the xlsx → data/report_parts_extracted.json)
python extraction/extract_report_parts.py
```

Python client pattern (same as `extraction/qa_extractor.py`):

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",   # or the compute node's hostname if cross-job
    api_key="token-abc123",                # vLLM accepts any string
)
response = client.chat.completions.create(
    model="Qwen3-VL-8B-Instruct",          # must match --model served name
    messages=[{"role": "user", "content": "..."}],
    temperature=0.0,
)
```

For the agent loop, see `baselines/run_agent.py` — it uses the same
OpenAI-compatible endpoint with guided decoding.

> **Cross-job access:** if the vLLM server runs in a *different* SLURM job than
> your client, replace `localhost` with the compute node's hostname (from
> `hostname` inside the server job) and ensure both jobs landed on the same node,
> or run client + server in the same `srun` session.

### 7.3 TITAN (retrieval encoder — not a chat server)

TITAN is a **frozen encoder**, not a generative API. You load weights from
`repos/TITAN` and run batch encoding jobs — no vLLM, no port.

Intended pipeline (WP2):

1. **Extract patches** from `.svs` slides (`openslide`, 512×512 tiles at ×10/×20).
2. **Encode patches** with the TITAN image encoder → cache per slide:
   `embeddings.pt` `[N×768]` + `coords.pt` `[N×2]`.
3. **Retrieve at inference** via `retrieval/titan_cosine.py` (cosine similarity
   between text query and patch embeddings).

Planned repo scripts (add under `scripts/cluster/` when ready):

```bash
# One-shot overnight batch — template pattern:
sbatch scripts/cluster/encode_titan.sh      # encodes all slides → dominik/embeddings/
```

Interactive smoke test on one slide:

```bash
srun --partition=24g --qos=students_normal --gres=gpu:1 --pty bash -l

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp dominik_mlmi
cd /mnt/projects/mlmi/reg2/repos/TITAN
# Follow TITAN repo README for encode API — wire output into TitanRetriever
python -c "from retrieval.titan_cosine import TitanCosineRetriever; print('OK')"
```

Cache embeddings under your personal work dir so teammates can share them:

```bash
mkdir -p /mnt/projects/mlmi/reg2/dominik/embeddings
chmod 777 /mnt/projects/mlmi/reg2/dominik/embeddings
```

See [`docs/PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) for the full TITAN wiring checklist.

### 7.4 Other deployed models (Patho-R1, InternVL, MedGemma)

Same general pattern applies:

| Step | Action |
|---|---|
| 1 | Find weights under `models/` or code under `repos/` |
| 2 | Start a SLURM job (`sbatch` or `srun`) — never on head |
| 3 | Run inside enroot with `/mnt` mounted |
| 4 | Record model path + port in `configs/paths.yaml` |
| 5 | Expose as an `AnswerBackend` in `agent/backends.py` or `baselines/run_agent.py` |

For HuggingFace-style models, the team container already has PyTorch + transformers.
For vLLM-served models, copy `start_qwen_server.sh` and point `qwen.model_path` in
`paths.yaml` at another entry under `models:` (+ adjust GPU count in the script).

### 7.5 SLURM script checklist (repo convention)

When adding `scripts/cluster/my_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=my-job
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --partition=24g
#SBATCH --qos=students_normal
#SBATCH --gres=gpu:1          # scale up for big models; CPU/RAM follow GPU count
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/my_job_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/my_job_%j.err

set -euo pipefail
# Use absolute path — sbatch copies the script to /var/spool/slurmd/…
source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths
mkdir -p "${LOGS_DIR}"

enroot start --root --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${CONTAINER}" \
  python "${REPO}/path/to/script.py" --arg value
```

Submit and monitor:

```bash
sbatch scripts/cluster/my_job.sh
squeue -u $USER
jobstats <job-id>
cat /mnt/projects/mlmi/reg2/dominik/logs/my_job_<job-id>.out
```

---

## 8. First SLURM Batch Job (WP1 — Data Exploration)

Script: `scripts/cluster/explore_data.sh`

```bash
# From head — submit from the repo (paths from configs/paths.yaml)
cd /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
sbatch scripts/cluster/explore_data.sh

# Monitor
squeue
jobstats <job-id>
cat /mnt/projects/mlmi/reg2/dominik/logs/explore_*.out
```

The batch script runs `notebooks/explore_wsi.ipynb` headlessly via `jupyter nbconvert`
inside the container set in `user.container_sqsh` (team `qwen25_dev_updated.sqsh` or your
`user.personal_container_sqsh` after you switch it in `paths.yaml`).

---

## 9. Quick Reference

| Task | Command |
|---|---|
| List containers / models | `ls /mnt/projects/mlmi/reg2/containers/` · `ls /mnt/projects/mlmi/reg2/models/` |
| First-time container | Import team base → `enroot create --name yourname_mlmi` → install deps → export (§3) |
| Interactive GPU | `srun --partition=24g --qos=students_normal --gres=gpu:1 --pty bash -l` |
| Cursor SSH job | `sbatch --partition=24g /mnt/general/examples/ssh.sh` |
| Start container | `enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp yourname_mlmi` |
| Export container | `enroot export --force --output .../yourname_$(date +%Y%m%d)_base.sqsh yourname_mlmi` |
| Git pull/push | `cd .../repos/mlmi_reg2_pathology_report_gen && git pull` |
| Start Qwen (vLLM) | `sbatch scripts/cluster/start_qwen_server.sh` |
| Test Qwen client | `python extraction/qa_extractor.py` |
| WP3 extraction | `python extraction/extract_report_parts.py` |
| Explore WSI (batch) | `sbatch scripts/cluster/explore_data.sh` |
| List models | `ls /mnt/projects/mlmi/reg2/models/` |
| Job queue | `squeue -u $USER` |
| GPU usage | `jobstats <job-id>` |
| Model config | `configs/paths.yaml` |
