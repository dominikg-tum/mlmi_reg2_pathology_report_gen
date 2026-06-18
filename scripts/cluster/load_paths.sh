# Source from SLURM scripts with an absolute path (sbatch copies scripts to /var/spool/slurmd/…).
# Sets REPO, CONTAINER, MODEL, MODEL_NAME, PROJECT_ROOT, LOGS_DIR from configs/paths.yaml.
#
# User secrets (HF_TOKEN, etc.):
#   1. Recommended: echo 'export HF_TOKEN=hf_...' > ~/.hf_env && chmod 600 ~/.hf_env
#   2. Or add export HF_TOKEN=... to ~/.bashrc (also parsed when bashrc skips non-interactive shells)

CLUSTER_REPO="/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen"

_cluster_paths_repo() {
  echo "${CLUSTER_REPO}"
}

# Load HF_TOKEN and other user exports for non-interactive SLURM jobs.
load_user_env() {
  if [[ -f "${HOME}/.hf_env" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/.hf_env"
  fi

  # ~/.bashrc often returns early for non-interactive shells — grep HF_TOKEN as fallback.
  if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.bashrc" ]]; then
    local token_line token
    token_line="$(grep -E '^[[:space:]]*(export[[:space:]]+)?HF_TOKEN=' "${HOME}/.bashrc" | tail -1 || true)"
    if [[ -n "${token_line}" ]]; then
      token="${token_line#*HF_TOKEN=}"
      token="${token#export }"
      token="${token#"${token%%[![:space:]]*}"}"
      token="${token%"${token##*[![:space:]]}"}"
      token="${token#\"}"; token="${token%\"}"
      token="${token#\'}"; token="${token%\'}"
      if [[ -n "${token}" ]]; then
        export HF_TOKEN="${token}"
      fi
    fi
  fi

  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
  fi
}

# Extra enroot args to forward HF auth into the container.
cluster_enroot_hf_env() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s\n' "--env" "HF_TOKEN=${HF_TOKEN}" "--env" "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}"
  fi
}

# Bash snippet run inside enroot before model download / encode jobs.
cluster_hf_login_snippet() {
  cat <<'EOF'
if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "${HF_TOKEN}" 2>/dev/null || true
fi
EOF
}

# MahmoodLab/TITAN pinned stack (HF model card requirements).
# Use container torch (CUDA 13 base); pin timm/einops/transformers — transformers>=5 breaks trust_remote_code.
cluster_offline_pip_snippet() {
  cat <<'EOF'
pip install -q timm==1.0.3 einops==0.6.1 einops-exts==0.0.4 transformers==4.46.0
pip install -q openslide-python pillow pyyaml tqdm huggingface_hub scikit-learn 2>/dev/null || true
EOF
}

# Lighter pip set for encode-only GPU jobs (same TITAN pins).
cluster_titan_pip_snippet() {
  cat <<'EOF'
pip install -q timm==1.0.3 einops==0.6.1 einops-exts==0.0.4 transformers==4.46.0
pip install -q openslide-python pillow pyyaml tqdm huggingface_hub 2>/dev/null || true
EOF
}

cluster_hybridrag_pip_snippet() {
  cat <<'EOF'
pip install -q pandas openpyxl pyyaml tqdm 2>/dev/null || true
pip install -q langchain-core langchain-community langchain-huggingface langchain-chroma langchain-classic rank-bm25 sentence-transformers 2>/dev/null || true
EOF
}

load_cluster_paths() {
  load_user_env

  local paths_yaml="${1:-$(_cluster_paths_repo)/configs/paths.yaml}"
  if [[ ! -f "${paths_yaml}" ]]; then
    echo "load_cluster_paths: missing ${paths_yaml}" >&2
    return 1
  fi
  if ! python3 -c "import yaml" 2>/dev/null; then
    echo "load_cluster_paths: PyYAML required (pip install pyyaml)" >&2
    return 1
  fi
  eval "$(python3 - "${paths_yaml}" <<'PY'
import sys
import yaml

p = yaml.safe_load(open(sys.argv[1]))
c, u, r, q = p["cluster"], p["user"], p["repo"], p["qwen"]

def emit(key, value):
    print(f'{key}="{value}"')

emit("PROJECT_ROOT", c["project_root"])
emit("REPO", r["path"])
emit("CONTAINER", u["container_sqsh"])
emit("LOGS_DIR", u["logs_dir"])
emit("MODEL", q["model_path"])
emit("MODEL_NAME", q["model_name"])
PY
)"
}
