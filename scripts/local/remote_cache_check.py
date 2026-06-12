#!/usr/bin/env python3
"""Run on cluster via SSH (file on pinned repo — no stdin piping over SSH).

Usage:
  python3 scripts/local/remote_cache_check.py REPO check INDEX   # exit 0 if complete
  python3 scripts/local/remote_cache_check.py REPO first         # print first incomplete index
"""

from __future__ import annotations

import sys
from pathlib import Path

REQUIRED = (
    "patch_embeddings_10x.pt",
    "patch_embeddings_20x.pt",
    "kmeans_centroids_10x.pt",
    "kmeans_centroids_20x.pt",
    "slide_embedding.pt",
)


def _load_context(repo: Path):
    sys.path.insert(0, str(repo))
    from scripts.vision._common import default_cache_root, default_data_dir, load_vision_config
    from vision.cache import slide_cache_dir
    from vision.wsi_io import find_svs_files, slide_id_from_path

    vcfg = load_vision_config()
    data_dir = default_data_dir()
    cache_root = default_cache_root(vcfg)
    files = find_svs_files(data_dir)
    return cache_root, files, slide_cache_dir, slide_id_from_path


def slide_complete(repo: Path, idx: int) -> bool:
    cache_root, files, slide_cache_dir, slide_id_from_path = _load_context(repo)
    if idx < 0 or idx >= len(files):
        return False
    out_dir = slide_cache_dir(cache_root, slide_id_from_path(files[idx]))
    return all((out_dir / name).exists() for name in REQUIRED)


def first_incomplete(repo: Path) -> int:
    cache_root, files, slide_cache_dir, slide_id_from_path = _load_context(repo)
    for idx, svs in enumerate(files):
        out_dir = slide_cache_dir(cache_root, slide_id_from_path(svs))
        if not all((out_dir / name).exists() for name in REQUIRED):
            return idx
    return len(files)


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__.strip(), file=sys.stderr)
        raise SystemExit(2)

    repo = Path(sys.argv[1])
    mode = sys.argv[2]
    arg = sys.argv[3]

    if mode == "check":
        raise SystemExit(0 if slide_complete(repo, int(arg)) else 1)
    if mode == "first":
        if arg != "-":
            print("third argument must be '-' for first mode", file=sys.stderr)
            raise SystemExit(2)
        print(first_incomplete(repo))
        raise SystemExit(0)

    print(f"unknown mode: {mode}", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
