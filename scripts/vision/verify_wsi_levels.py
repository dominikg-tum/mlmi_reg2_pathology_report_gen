"""Inspect one WSI pyramid: level_downsamples, mpp-x, and tiling objective mapping.

Run on a compute node (not the cluster head). Lightweight read-only openslide metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.vision._common import default_data_dir, load_paths_config
from vision.wsi_io import (
    _MPP_BY_OBJECTIVE,
    find_svs_files,
    objective_downsample,
    slide_mpp_x,
)

# Fallback to verify when openslide.mpp-x is absent (×4 objective ≈ 2.5 µm/px).
_MPP_4X_FALLBACK = 2.5


def _resolve_slide(data_dir: Path, slide_arg: str) -> Path:
    if slide_arg:
        candidates = [data_dir / slide_arg]
        if not candidates[0].exists():
            candidates = list(data_dir.rglob(slide_arg))
        if not candidates:
            raise SystemExit(f"No slide matching {slide_arg!r} under {data_dir}")
        return candidates[0]

    files = find_svs_files(data_dir, limit=1)
    if not files:
        raise SystemExit(f"No .svs files under {data_dir}")
    return files[0]


def _approx_mag_at_level(slide_mpp: float, level_downsample: float, ref_mpp_20x: float = 0.5) -> float:
    """Rough magnification vs a 20× reference (0.5 µm/px at level 0)."""
    effective_mpp = slide_mpp * level_downsample
    return ref_mpp_20x / effective_mpp * 20.0


def inspect_slide(svs_path: Path) -> None:
    import openslide

    slide = openslide.OpenSlide(str(svs_path))
    try:
        print("=" * 72)
        print("WSI pyramid check (use these levels — do not invent magnifications)")
        print("=" * 72)
        print(f"slide:     {svs_path}")
        print(f"size (L0): {slide.dimensions[0]} x {slide.dimensions[1]} px")
        print(f"levels:    {slide.level_count}")
        print()

        downsamples = tuple(float(x) for x in slide.level_downsamples)
        print("level_downsamples:", downsamples)
        print("level_dimensions:", slide.level_dimensions)
        print()

        props = dict(slide.properties)
        mpp_raw = props.get("openslide.mpp-x")
        mpp_y = props.get("openslide.mpp-y")
        print("openslide.mpp-x:", mpp_raw if mpp_raw is not None else "<missing>")
        print("openslide.mpp-y:", mpp_y if mpp_y is not None else "<missing>")
        vendor = props.get("openslide.vendor", "<unknown>")
        objective = props.get("openslide.objective-power", "<unknown>")
        print("vendor:            ", vendor)
        print("objective-power:   ", objective)
        print()

        slide_mpp = slide_mpp_x(slide)
        print("slide_mpp_x() used by vision/wsi_io.py:", slide_mpp)
        if mpp_raw is None:
            print(
                f"  (no mpp-x property — wsi_io falls back to 0.25 µm/px, "
                f"not _MPP_BY_OBJECTIVE)"
            )
        print()

        print("Per-level approximate magnification (vs 20× @ 0.5 µm/px at L0):")
        for i, ds in enumerate(downsamples):
            mag = _approx_mag_at_level(slide_mpp, ds)
            w, h = slide.level_dimensions[i]
            print(f"  level {i}: downsample={ds:6.1f}  ~{mag:5.1f}×  dims={w} x {h}")
        print()

        print("×4 MPP fallback check (_MPP_BY_OBJECTIVE['4x'] = 2.5 µm/px):")
        if mpp_raw is None:
            assumed_mpp = _MPP_4X_FALLBACK
            print(f"  mpp-x missing → using fallback {assumed_mpp} µm/px for ×4 objective")
        else:
            assumed_mpp = float(mpp_raw)
            print(f"  mpp-x present ({assumed_mpp} µm/px) — fallback not needed")
        downsample_4x = _MPP_4X_FALLBACK / assumed_mpp
        level_4x = slide.get_best_level_for_downsample(downsample_4x)
        level_ds_4x = float(slide.level_downsamples[level_4x])
        print(f"  target downsample for ×4: {downsample_4x:.4f}")
        print(f"  get_best_level_for_downsample → level {level_4x} (downsample={level_ds_4x})")
        print()

        print("vision/wsi_io.py objective mapping (_MPP_BY_OBJECTIVE):")
        for obj, target_mpp in sorted(_MPP_BY_OBJECTIVE.items()):
            ds = objective_downsample(slide, obj)
            level = slide.get_best_level_for_downsample(ds)
            level_ds = float(slide.level_downsamples[level])
            print(
                f"  {obj:>3}: target_mpp={target_mpp}  "
                f"downsample={ds:7.3f}  → level {level} (actual ds={level_ds})"
            )
        print()
        print("Use the level_downsamples tuple above when choosing pyramid levels.")
    finally:
        slide.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect WSI pyramid levels and MPP/objective mapping for one slide."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="WSI root directory (default: configs/paths.yaml cluster.data_dir)",
    )
    parser.add_argument(
        "--slide",
        type=str,
        default="",
        help="Single slide filename or path fragment (default: first .svs found)",
    )
    args = parser.parse_args()

    cfg = load_paths_config()
    data_dir = args.data_dir or default_data_dir(cfg)
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    svs_path = _resolve_slide(data_dir, args.slide.strip())
    inspect_slide(svs_path)


if __name__ == "__main__":
    main()
