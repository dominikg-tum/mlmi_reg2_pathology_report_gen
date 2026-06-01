# Source from SLURM scripts: source "${SCRIPT_DIR}/load_paths.sh"
# Sets REPO, CONTAINER, MODEL, MODEL_NAME, PROJECT_ROOT, LOGS_DIR from configs/paths.yaml.

_cluster_paths_repo() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  echo "$(cd "${script_dir}/../.." && pwd)"
}

load_cluster_paths() {
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
