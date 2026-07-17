"""Unified offline WSI preprocessing: tile → verify → CONCH encode → kmeans (optional)."""

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
from vision.cache import slide_cache_dir
from vision.wsi_io import resolve_wsi_files, slide_id_from_path


def _validate_artifacts(
    cache_root: Path,
    svs_files: list[Path],
    *,
    levels: list[str] | None = None,
    require_slide_embedding: bool = False,
) -> None:
    """Exit non-zero if required encode_levels patch embeddings are missing."""
    levels = levels or default_encode_levels()
    missing: list[str] = []
    for svs_path in svs_files:
        sid = slide_id_from_path(svs_path)
        out_dir = slide_cache_dir(cache_root, sid)
        for lv in levels:
            emb = f"patch_embeddings_{lv}.pt"
            if not (out_dir / emb).exists():
                missing.append(f"{sid}/{emb}")
        if require_slide_embedding and not (out_dir / "slide_embedding.pt").exists():
            missing.append(f"{sid}/slide_embedding.pt")
    if missing:
        raise SystemExit(
            "Offline pipeline finished but required artifacts are missing:\n  "
            + "\n  ".join(missing)
        )


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
        description="Run offline WSI pipeline (tile → verify → CONCH encode → kmeans)."
    )
    parser.add_argument("--slide", type=str, default="")
    parser.add_argument("--wsi-index", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--skip-tile", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--skip-kmeans", action="store_true")
    parser.add_argument(
        "--with-slide-emb",
        action="store_true",
        help="Also run TITAN slide_embedding.pt (Phase 2 / ablation; not needed for CONCH retrieve)",
    )
    parser.add_argument(
        "--skip-slide-emb",
        action="store_true",
        help=argparse.SUPPRESS,
    )
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

    run_slide_emb = args.with_slide_emb and not args.skip_slide_emb
    if run_slide_emb:
        _run_module("scripts.vision.encode_slide_embeddings", common)

    svs_files = resolve_wsi_files(
        data_dir, slide=args.slide, wsi_index=args.wsi_index
    )
    _validate_artifacts(
        cache_root,
        svs_files,
        levels=levels,
        require_slide_embedding=run_slide_emb,
    )
    print(f"Offline pipeline complete -> {cache_root}")


if __name__ == "__main__":
    main()
