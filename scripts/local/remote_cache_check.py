#!/usr/bin/env python3
"""Run on cluster via SSH (file on pinned repo — no stdin piping over SSH).

Usage:
  python3 scripts/local/remote_cache_check.py REPO check INDEX           # exit 0 if complete
  python3 scripts/local/remote_cache_check.py REPO first -               # print first incomplete index
  python3 scripts/local/remote_cache_check.py REPO incomplete START END  # print incomplete indices in [S,E]
  python3 scripts/local/remote_cache_check.py REPO ensure-manifest -     # build/load manifest only
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# 20x-only corpus (configs/vision.yaml encode_levels). K-means is optional ablation.
REQUIRED = (
    "patch_embeddings_20x.pt",
)

MANIFEST_NAME = "wsi_svs_index_manifest.txt"


def _load_repo(repo: Path):
    sys.path.insert(0, str(repo))
    from scripts.vision._common import default_cache_root, load_vision_config
    from vision.cache import slide_cache_dir
    from vision.wsi_io import find_svs_files, slide_id_from_path

    vcfg = load_vision_config()
    cache_root = default_cache_root(vcfg)
    return vcfg, cache_root, slide_cache_dir, find_svs_files, slide_id_from_path


def _write_manifest(manifest: Path, ids: list[str]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(ids) + "\n")


def _slide_ids_from_name_map(repo: Path) -> list[str] | None:
    from vision.wsi_mapping import mapped_slide_ids, name_map_available

    csv_path = str(repo / "data" / "manifests" / "wsi_name_map.csv")
    if not name_map_available(csv_path):
        return None
    ids = mapped_slide_ids(csv_path=csv_path)
    return ids or None


def _slide_ids_from_thumbnails(vcfg: dict) -> list[str] | None:
    from vision.cache import dataset_thumbnail_dir

    bank = dataset_thumbnail_dir(vcfg)
    if bank is None or not bank.is_dir():
        return None
    stems = sorted(path.stem for path in bank.glob("*.jpg"))
    if not stems:
        return None
    return [f"{stem}.svs" for stem in stems]


def _slide_ids_from_find(data_dir: Path, find_svs_files, slide_id_from_path) -> list[str]:
    print(f"Scanning {data_dir} for .svs via find(1) (one-time)...", file=sys.stderr, flush=True)
    proc = subprocess.run(
        ["find", str(data_dir), "-name", "*.svs", "-type", "f"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        print("find failed; falling back to Python rglob...", file=sys.stderr, flush=True)
        return [slide_id_from_path(path) for path in find_svs_files(data_dir)]
    paths = sorted(Path(line.strip()) for line in proc.stdout.splitlines() if line.strip())
    print(f"Found {len(paths)} .svs files.", file=sys.stderr, flush=True)
    return [slide_id_from_path(path) for path in paths]


def _load_slide_ids(repo: Path) -> list[str]:
    vcfg, cache_root, _, find_svs_files, slide_id_from_path = _load_repo(repo)
    manifest = cache_root / MANIFEST_NAME
    if manifest.is_file():
        ids = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
        if ids:
            return ids

    ids = _slide_ids_from_name_map(repo)
    if ids:
        print(
            f"Built wsi index manifest from wsi_name_map.csv ({len(ids)} slides).",
            file=sys.stderr,
            flush=True,
        )
        _write_manifest(manifest, ids)
        return ids

    ids = _slide_ids_from_thumbnails(vcfg)
    if ids:
        print(
            f"Built wsi index manifest from flat thumbnail bank ({len(ids)} slides).",
            file=sys.stderr,
            flush=True,
        )
        _write_manifest(manifest, ids)
        return ids

    from scripts.vision._common import default_data_dir

    ids = _slide_ids_from_find(default_data_dir(), find_svs_files, slide_id_from_path)
    _write_manifest(manifest, ids)
    print(f"Wrote {len(ids)} slide ids to {manifest}", file=sys.stderr, flush=True)
    return ids


def ensure_manifest(repo: Path) -> int:
    ids = _load_slide_ids(repo)
    print(len(ids))
    return len(ids)


def _artifacts_complete(cache_root: Path, slide_cache_dir, slide_id: str) -> bool:
    out_dir = slide_cache_dir(cache_root, slide_id)
    return all((out_dir / name).exists() for name in REQUIRED)


def slide_complete(repo: Path, idx: int) -> bool:
    _, cache_root, slide_cache_dir, _, _ = _load_repo(repo)
    slide_ids = _load_slide_ids(repo)
    if idx < 0 or idx >= len(slide_ids):
        return False
    return _artifacts_complete(cache_root, slide_cache_dir, slide_ids[idx])


def first_incomplete(repo: Path) -> int:
    _, cache_root, slide_cache_dir, _, _ = _load_repo(repo)
    slide_ids = _load_slide_ids(repo)
    for idx, slide_id in enumerate(slide_ids):
        if not _artifacts_complete(cache_root, slide_cache_dir, slide_id):
            return idx
    return len(slide_ids)


def incomplete_in_range(repo: Path, start: int, end: int) -> list[int]:
    _, cache_root, slide_cache_dir, _, _ = _load_repo(repo)
    slide_ids = _load_slide_ids(repo)
    hi = min(end, len(slide_ids) - 1)
    if start > hi:
        return []
    out: list[int] = []
    for idx in range(start, hi + 1):
        if not _artifacts_complete(cache_root, slide_cache_dir, slide_ids[idx]):
            out.append(idx)
    return out


def main() -> None:
    if len(sys.argv) not in (4, 5):
        print(__doc__.strip(), file=sys.stderr)
        raise SystemExit(2)

    repo = Path(sys.argv[1])
    mode = sys.argv[2]

    if mode == "check":
        if len(sys.argv) != 4:
            print("check mode: python3 ... REPO check INDEX", file=sys.stderr)
            raise SystemExit(2)
        raise SystemExit(0 if slide_complete(repo, int(sys.argv[3])) else 1)

    if mode == "first":
        if len(sys.argv) != 4 or sys.argv[3] != "-":
            print("first mode: python3 ... REPO first -", file=sys.stderr)
            raise SystemExit(2)
        print(first_incomplete(repo))
        raise SystemExit(0)

    if mode == "incomplete":
        if len(sys.argv) != 5:
            print("incomplete mode: python3 ... REPO incomplete START END", file=sys.stderr)
            raise SystemExit(2)
        start = int(sys.argv[3])
        end = int(sys.argv[4])
        for idx in incomplete_in_range(repo, start, end):
            print(idx)
        raise SystemExit(0)

    if mode == "ensure-manifest":
        if len(sys.argv) != 4 or sys.argv[3] != "-":
            print("ensure-manifest mode: python3 ... REPO ensure-manifest -", file=sys.stderr)
            raise SystemExit(2)
        ensure_manifest(repo)
        raise SystemExit(0)

    print(f"unknown mode: {mode}", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
