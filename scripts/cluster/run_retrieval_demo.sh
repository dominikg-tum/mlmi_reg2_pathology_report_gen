#!/bin/bash
#SBATCH --job-name=retrieval-demo-smoke
#SBATCH --chdir=/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=ALL
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/mlmi/reg2/dominik/logs/retrieval_demo_%j.out
#SBATCH --error=/mnt/projects/mlmi/reg2/dominik/logs/retrieval_demo_%j.err

set -euo pipefail

WSI_PATH="${WSI_PATH:-}"
QUESTIONS="${QUESTIONS:-all}"

# shellcheck source=load_paths.sh
source /mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/scripts/cluster/load_paths.sh
load_cluster_paths

if [[ -z "${WSI_PATH}" ]]; then
  WSI_PATH="$(python3 - <<'PY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(open("/mnt/projects/mlmi/reg2/repos/mlmi_reg2_pathology_report_gen/configs/paths.yaml"))
data_dir = Path(cfg["cluster"]["data_dir"])
hits = list(data_dir.rglob("TUM_Uterus_0001.svs"))
if not hits:
    raise SystemExit("TUM_Uterus_0001.svs not found under data_dir")
print(hits[0])
PY
)"
fi

mkdir -p "${LOGS_DIR}"

CONTAINER="${PERSONAL_CONTAINER:-/mnt/projects/mlmi/reg2/containers/dominik_20260529_base.sqsh}"
readarray -t _ENROOT_HF_ENV < <(cluster_enroot_hf_env)

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${_ENROOT_HF_ENV[@]}" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
$(cluster_titan_pip_snippet)
    run_demo() {
      python -m scripts.vision.run_retrieval_demo \
        --wsi-path '${WSI_PATH}' \
        --question \"\$1\" \
        \"\${@:2}\"
    }
    if [[ '${QUESTIONS}' == 'all' ]]; then
      run_demo 'irregular crowded endometrial glands with cytologic atypia' --node-tier local_features --k 5
      run_demo 'whorled smooth muscle fascicles' --node-tier local_features --k 5
      run_demo 'integrate findings into final diagnosis' --node-tier integration --integration --k 5
    else
      run_demo '${QUESTIONS}' --node-tier local_features --k 5
    fi
  "
