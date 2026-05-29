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
| `containers/` | Pre-built `.sqsh` container images (`graphrag_*` = newest team base) |
| `repos/` | Git repositories (clone yours here) |
| `models/` | Locally deployed VLMs (Qwen lives here) |
| `scripts/` | Shared helper scripts |
| `case_reports_to_korea_collaborators.xlsx` | Labels (`slide_id`, `case_class`, `english_report`) |
| `requirements.txt` / `requirements_clean.txt` | Dependencies to install inside containers |
| `dominik/` | Your personal working directory (create once, `chmod 777`) |

This repository should be cloned to:

```text
/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
```

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
# Import and create from the newest team container (graphrag = latest)
enroot import -- /mnt/projects/mlmi/reg2/containers/graphrag_latest.sqsh
enroot create --name my_env graphrag_latest.sqsh
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

### 3.3 Install dependencies (inside the container)

```bash
pip list
cat /mnt/projects/mlmi/reg2/requirements_clean.txt

pip install -r /mnt/projects/mlmi/reg2/requirements_clean.txt
pip install -r /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/requirements.txt

# WP1 / WP2 extras
pip install openslide-python opencv-python-headless tqdm

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
# From head — 24G partition, one GPU, 32G RAM
srun --partition=24g --qos=students_normal --gres=gpu:1 --mem=32G --pty bash -l
```

### 4.2 Create your working directory

```bash
mkdir -p /mnt/projects/mlmi/reg2/dominik/logs
chmod 777 /mnt/projects/mlmi/reg2/dominik
```

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

---

## 7. Q&A Extraction with Local Qwen (WP3)

No public API on the cluster — use the locally deployed Qwen model under `models/`.

```bash
ls /mnt/projects/mlmi/reg2/models/
ls /mnt/projects/mlmi/reg2/repos/
grep -i qwen /mnt/projects/mlmi/reg2/scripts/*.sh
```

### Scenario A: Qwen served via vLLM (OpenAI-compatible API)

See `src/wp3_qa_extraction/extract_qa.py` — point `base_url` at the local endpoint (check port with teammates):

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

response = client.chat.completions.create(
    model="Qwen2.5-72B-Instruct",  # confirm exact model name
    messages=[...],
    temperature=0.0,
)
```

### Scenario B: Start a vLLM server yourself

Submit `scripts/cluster/start_qwen_server.sh`:

```bash
sbatch scripts/cluster/start_qwen_server.sh
```

Or from the cluster scripts folder after copying/adapting paths.

---

## 8. First SLURM Batch Job (WP1 — Data Exploration)

Script: `scripts/cluster/explore_data.sh`

```bash
# From head
sbatch /mnt/projects/mlmi/reg2/dominik/explore_data.sh
# or, from the repo:
sbatch scripts/cluster/explore_data.sh

# Monitor
squeue
jobstats <job-id>
cat /mnt/projects/mlmi/reg2/dominik/logs/explore_*.out
```

The batch script runs `explore_wsi.py` inside `dominik_mlmi` (or your latest exported `.sqsh`).

---

## 9. Quick Reference

| Task | Command |
|---|---|
| Interactive GPU | `srun --partition=24g --qos=students_normal --gres=gpu:1 --mem=32G --pty bash -l` |
| Start container | `enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp dominik_mlmi` |
| Export container | `enroot export --force --output .../dominik_$(date +%Y%m%d)_base.sqsh dominik_mlmi` |
| Submit job | `sbatch scripts/cluster/explore_data.sh` |
| Job queue | `squeue` |
| GPU usage | `jobstats <job-id>` |
