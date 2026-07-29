#!/bin/bash
#SBATCH --job-name=pathology-build-hybridrag
#SBATCH --chdir=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen
#SBATCH --export=NONE
#SBATCH --partition=24g
#SBATCH --qos=students_opportunistic
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/logs/build_hybridrag_%j.out
#SBATCH --error=/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/logs/build_hybridrag_%j.err
#
# Build HybridRAG Chroma + BM25 index (train reports ± CAP reference chunks).
#
# Usage:
#   # reports-only (baseline b2)
#   VARIANT=nocap FORCE_REBUILD=1 sbatch --export=NONE,VARIANT,FORCE_REBUILD \
#     --job-name=path-hybridrag-nocap scripts/cluster/build_hybridrag_index.sh
#
#   # reports + CAP/WHO refs (baseline b2_cap)
#   VARIANT=cap FORCE_REBUILD=1 sbatch --export=NONE,VARIANT,FORCE_REBUILD \
#     --job-name=path-hybridrag-cap scripts/cluster/build_hybridrag_index.sh
#
#   # both ablation stores in one job
#   VARIANT=both FORCE_REBUILD=1 sbatch --export=NONE,VARIANT,FORCE_REBUILD \
#     --job-name=path-hybridrag-both scripts/cluster/build_hybridrag_index.sh
#
# Coordinate before FORCE_REBUILD on shared chroma paths (Nick).

set -euo pipefail

export PATH="/usr/local/bin:/usr/bin:/bin${PATH:+:${PATH}}"

PINNED_REPO="${MLMI_PINNED_REPO:-/mnt/projects/mlmi/TUMUntera/dominik_garstenauer/repos/mlmi_reg2_pathology_report_gen}"
export MLMI_PINNED_REPO="${PINNED_REPO}"

SPLIT="${SPLIT:-train}"
VARIANT="${VARIANT:-nocap}"
FORCE_REBUILD="${FORCE_REBUILD:-}"

# shellcheck source=load_paths.sh
source "${PINNED_REPO}/scripts/cluster/load_paths.sh"
load_cluster_paths "${PINNED_REPO}/configs/paths.yaml"
REPO="${PINNED_REPO}"

mkdir -p "${LOGS_DIR}"

CONTAINER="${PERSONAL_CONTAINER:-${CONTAINER}}"

enroot start --rw --mount /mnt:/mnt --mount /tmp:/tmp \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    cd '${REPO}'
$(cluster_hybridrag_pip_snippet)
    build_one() {
      local variant=\"\$1\"
      local args=(python -m scripts.memory.build_hybridrag_index --split '${SPLIT}' --variant \"\${variant}\")
      [[ -n '${FORCE_REBUILD}' ]] && args+=(--force-rebuild)
      echo \"Building HybridRAG variant=\${variant} ...\"
      \"\${args[@]}\"
    }
    case '${VARIANT}' in
      both)
        build_one nocap
        build_one cap
        ;;
      nocap|cap)
        build_one '${VARIANT}'
        ;;
      *)
        echo \"Unknown VARIANT=${VARIANT}; use nocap|cap|both\" >&2
        exit 2
        ;;
    esac
  "
