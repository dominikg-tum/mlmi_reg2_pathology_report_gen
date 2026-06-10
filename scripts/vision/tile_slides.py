"""Offline: tile WSI at configured magnification bands → coords_{level}.pt (CPU only)."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scripts.vision._common import (
    default_cache_root,
    default_data_dir,
    default_encode_levels,
    load_vision_config,
    slide_log_path,
)
from vision.cache import slide_cache_dir
from vision.mag_config import VALID_ZOOM_LEVELS
from vision.patching import extract_patch_coords
from vision.wsi_io import resolve_wsi_files, slide_id_from_path


def _tile_one(
    svs_path: Path,
    out_dir: Path,
    level: str,
    *,
    max_patches: int,
) -> None:
    coord_path = out_dir / f"coords_{level}.pt"
    meta_path = out_dir / f"meta_{level}.json"
    if coord_path.exists() and meta_path.exists():
        print(f"SKIP {slide_id_from_path(svs_path)} {level} (coords exist)")
        return

    coords, patch_size_lv0 = extract_patch_coords(
        svs_path, mag_band=level, max_patches=max_patches
    )
    if not coords:
        raise RuntimeError("no tissue patches")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(np.asarray(coords, dtype=np.int64), coord_path)
    meta = {
        "slide_id": slide_id_from_path(svs_path),
        "level": level,
        "n_patches": len(coords),
        "patch_size_lv0": patch_size_lv0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"OK  {slide_id_from_path(svs_path)} {level} n={len(coords)} -> {coord_path}")


def _check_only(cache_root: Path, data_dir: Path, levels: list[str]) -> None:
    files = resolve_wsi_files(data_dir)
    done, failed, missing = 0, 0, 0
    for svs_path in files:
        sid = slide_id_from_path(svs_path)
        out_dir = slide_cache_dir(cache_root, sid)
        for level in levels:
            done_path = slide_log_path(cache_root, sid, f"tiled_{level}.done")
            fail_path = slide_log_path(cache_root, sid, f"tiled_{level}.failed")
            coord_path = out_dir / f"coords_{level}.pt"
            if done_path.exists() or coord_path.exists():
                done += 1
            elif fail_path.exists():
                failed += 1
            else:
                missing += 1
    total = len(files) * len(levels)
    print(f"Tiling status: {done} done / {failed} failed / {missing} missing (of {total} tasks)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tile WSIs and save patch coordinates only.")
    parser.add_argument("--level", choices=list(VALID_ZOOM_LEVELS), action="append")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--slide", type=str, default="")
    parser.add_argument("--wsi-index", type=int, default=None)
    parser.add_argument("--max-patches", type=int, default=0, help="0 = no cap")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    vcfg = load_vision_config()
    data_dir = args.data_dir or default_data_dir()
    cache_root = args.cache_root or default_cache_root(vcfg)
    levels = args.level or default_encode_levels()

    if args.check_only:
        _check_only(cache_root, data_dir, levels)
        return

    svs_files = resolve_wsi_files(
        data_dir, slide=args.slide, limit=args.limit, wsi_index=args.wsi_index
    )
    if not svs_files:
        raise SystemExit(f"No .svs files under {data_dir}")

    (cache_root / "logs").mkdir(parents=True, exist_ok=True)
    ok, failed = 0, 0
    for svs_path in svs_files:
        sid = slide_id_from_path(svs_path)
        out_dir = slide_cache_dir(cache_root, sid)
        for level in levels:
            done_path = slide_log_path(cache_root, sid, f"tiled_{level}.done")
            fail_path = slide_log_path(cache_root, sid, f"tiled_{level}.failed")
            try:
                _tile_one(
                    svs_path,
                    out_dir,
                    level,
                    max_patches=args.max_patches,
                )
                done_path.write_text(datetime.now(timezone.utc).isoformat() + "\n")
                if fail_path.exists():
                    fail_path.unlink()
                ok += 1
            except Exception:
                fail_path.write_text(traceback.format_exc())
                print(f"FAIL {sid} {level}: see {fail_path}")
                failed += 1

    print(f"Done: {ok} ok, {failed} failed")


if __name__ == "__main__":
    main()
