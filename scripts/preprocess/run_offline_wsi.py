"""Unified offline WSI preprocessing: tile → verify → encode → kmeans → slide emb."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.vision._common import (
    default_cache_root,
    default_data_dir,
    default_encode_levels,
    load_vision_config,
)
from vision.wsi_io import resolve_wsi_files


def _run_module(module: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *extra_args]
    print(f">>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _common_args(
    cache_root: Path,
    data_dir: Path,
    slide: str,
    wsi_index: int | None,
) -> list[str]:
    args = ["--cache-root", str(cache_root), "--data-dir", str(data_dir)]
    if slide:
        args.extend(["--slide", slide])
    elif wsi_index is not None:
        args.extend(["--wsi-index", str(wsi_index)])
    return args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full offline WSI pipeline (tile → verify → encode → kmeans → slide emb)."
    )
    parser.add_argument("--slide", type=str, default="")
    parser.add_argument("--wsi-index", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--skip-tile", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--skip-kmeans", action="store_true")
    parser.add_argument("--skip-slide-emb", action="store_true")
    parser.add_argument("--verify-level", choices=["5x", "10x", "20x", "40x"], default="20x")
    args = parser.parse_args()

    if not args.slide and args.wsi_index is None:
        raise SystemExit("Provide --slide or --wsi-index")

    vcfg = load_vision_config()
    data_dir = args.data_dir or default_data_dir()
    cache_root = args.cache_root or default_cache_root(vcfg)
    levels = default_encode_levels()
    level_args: list[str] = []
    for lv in levels:
        level_args.extend(["--level", lv])

    common = _common_args(cache_root, data_dir, args.slide, args.wsi_index)

    if not args.skip_tile:
        _run_module("scripts.vision.tile_slides", common + level_args)

    if not args.skip_verify:
        svs_files = resolve_wsi_files(
            data_dir, slide=args.slide, wsi_index=args.wsi_index
        )
        for svs_path in svs_files:
            _run_module(
                "scripts.vision.verify_tiling",
                [
                    "--wsi-path",
                    str(svs_path),
                    "--cache-root",
                    str(cache_root),
                    "--level",
                    args.verify_level,
                ],
            )

    if not args.skip_encode:
        _run_module("scripts.vision.encode_patches_offline", common + level_args)

    if not args.skip_kmeans:
        _run_module("scripts.vision.build_kmeans_index", common + level_args)

    if not args.skip_slide_emb:
        _run_module("scripts.vision.encode_slide_embeddings", common)

    print(f"Offline pipeline complete -> {cache_root}")


if __name__ == "__main__":
    main()
