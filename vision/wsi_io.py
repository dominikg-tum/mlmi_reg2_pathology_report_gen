"""Shared openslide helpers for offline WSI jobs (never called during agent inference)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import numpy as np

# Approximate microns-per-pixel at common objectives (fallback when mpp-x missing).
_MPP_BY_OBJECTIVE = {"5x": 2.0, "4x": 2.5, "10x": 1.0, "20x": 0.5, "40x": 0.25}


def slide_id_from_path(svs_path: Path) -> str:
    """Canonical cache key: TUM_Uterus_XXXX.svs when mapped, else basename."""
    from vision.wsi_mapping import canonical_slide_id

    return canonical_slide_id(svs_path)


def find_svs_files(data_dir: Path, *, limit: int = 0) -> list[Path]:
    """List .svs paths. Prefers name-map order (464 TUM_Uterus ids) when CSV exists."""
    from vision.wsi_mapping import resolve_mapped_wsi_files

    mapped = resolve_mapped_wsi_files(data_dir, limit=limit)
    if mapped is not None:
        return mapped
    files = sorted(data_dir.rglob("*.svs"))
    if limit > 0:
        files = files[:limit]
    return files


def resolve_wsi_files(
    data_dir: Path,
    *,
    slide: str = "",
    limit: int = 0,
    wsi_index: int | None = None,
) -> list[Path]:
    """Resolve slide list; optional SLURM array index into the name map (not raw FS sort)."""
    from vision.wsi_mapping import resolve_mapped_wsi_files

    mapped = resolve_mapped_wsi_files(
        data_dir, slide=slide, limit=limit, wsi_index=wsi_index
    )
    if mapped is not None:
        return mapped

    if slide:
        files = sorted(data_dir.rglob(slide))
    else:
        files = sorted(data_dir.rglob("*.svs"))
        if limit > 0:
            files = files[:limit]
    if wsi_index is not None:
        if wsi_index < 0 or wsi_index >= len(files):
            raise IndexError(
                f"wsi_index={wsi_index} out of range for {len(files)} slides under {data_dir}"
            )
        return [files[wsi_index]]
    return files


def _target_mpp(objective: str) -> float:
    key = objective.lower().replace("×", "x").strip()
    if key not in _MPP_BY_OBJECTIVE:
        raise ValueError(f"Unknown objective {objective!r}; use one of {list(_MPP_BY_OBJECTIVE)}")
    return _MPP_BY_OBJECTIVE[key]


def slide_mpp_x(slide) -> float:
    raw = slide.properties.get("openslide.mpp-x")
    if raw is not None:
        return float(raw)
    return 0.25


def objective_downsample(slide, objective: str) -> float:
    """Openslide downsample factor to read near the requested objective."""
    return _target_mpp(objective) / slide_mpp_x(slide)


def _read_level_for_objective(slide, objective: str, patch_size: int) -> tuple[int, float, int]:
    downsample = objective_downsample(slide, objective)
    level = slide.get_best_level_for_downsample(downsample)
    level_downsample = float(slide.level_downsamples[level])
    patch_size_lv0 = int(round(patch_size * level_downsample))
    return level, level_downsample, patch_size_lv0


def is_tissue_patch(arr: np.ndarray, background_threshold: int = 220) -> bool:
    """Drop glass/background tiles (grayscale mean above threshold)."""
    if arr.ndim == 3:
        gray = arr.mean(axis=2)
    else:
        gray = arr
    return float(gray.mean()) <= background_threshold


def write_thumbnail(svs_path: Path, out_path: Path, *, max_edge_px: int = 1024) -> None:
    """P1 baseline: native pyramid downsample → blurry whole-slide thumbnail."""
    import openslide

    slide = openslide.OpenSlide(str(svs_path))
    try:
        thumb = slide.get_thumbnail((max_edge_px, max_edge_px)).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        thumb.save(out_path)
    finally:
        slide.close()


def load_patch_at_coord(
    svs_path: Path,
    coord_level0: tuple[int, int],
    *,
    objective: str = "20x",
    patch_size: int = 512,
):
    """Read one RGB patch at a level-0 top-left corner."""
    import openslide

    slide = openslide.OpenSlide(str(svs_path))
    try:
        level, level_downsample, _ = _read_level_for_objective(slide, objective, patch_size)
        x0, y0 = coord_level0
        region = slide.read_region((int(x0), int(y0)), level, (patch_size, patch_size)).convert(
            "RGB"
        )
        return region
    finally:
        slide.close()


def _zoom_crop_topleft(
    coord_level0: tuple[int, int],
    *,
    from_patch_size_lv0: int,
    to_patch_size_lv0: int,
) -> tuple[int, int]:
    cx = int(coord_level0[0] + from_patch_size_lv0 / 2.0)
    cy = int(coord_level0[1] + from_patch_size_lv0 / 2.0)
    x1 = int(round(cx - to_patch_size_lv0 / 2.0))
    y1 = int(round(cy - to_patch_size_lv0 / 2.0))
    return x1, y1


def zoom_crop_at_coord(
    svs_path: Path,
    coord_level0: tuple[int, int],
    *,
    from_zoom: str = "20x",
    to_zoom: str = "40x",
):
    """Crop a to_zoom tile centered on the from_zoom tile center (level-0 coords)."""
    import openslide

    from vision.mag_config import clamp_runtime_zoom, zoom_config

    from_zoom = clamp_runtime_zoom(from_zoom)
    to_zoom = clamp_runtime_zoom(to_zoom)

    from_objective, from_patch_size, _ = zoom_config(from_zoom)
    to_objective, to_patch_size, _ = zoom_config(to_zoom)

    slide = openslide.OpenSlide(str(svs_path))
    try:
        _, _, from_ps_lv0 = _read_level_for_objective(slide, from_objective, from_patch_size)
        _, _, to_ps_lv0 = _read_level_for_objective(slide, to_objective, to_patch_size)

        x1, y1 = _zoom_crop_topleft(
            coord_level0,
            from_patch_size_lv0=from_ps_lv0,
            to_patch_size_lv0=to_ps_lv0,
        )

        w0, h0 = slide.dimensions
        x1 = max(0, min(int(x1), max(w0 - to_ps_lv0, 0)))
        y1 = max(0, min(int(y1), max(h0 - to_ps_lv0, 0)))

        return load_patch_at_coord(
            svs_path,
            (x1, y1),
            objective=to_objective,
            patch_size=to_patch_size,
        )
    finally:
        slide.close()


def find_parent_patch_index(
    coord_high: tuple[int, int],
    coords_medium: np.ndarray,
    *,
    patch_size_lv0_high: int,
    patch_size_lv0_medium: int,
) -> int:
    """Index into coords_medium for the ×10 tile containing the ×20 patch centre."""
    cx = coord_high[0] + patch_size_lv0_high // 2
    cy = coord_high[1] + patch_size_lv0_high // 2
    for i, (x, y) in enumerate(coords_medium):
        if x <= cx < x + patch_size_lv0_medium and y <= cy < y + patch_size_lv0_medium:
            return int(i)
    centres = coords_medium.astype(np.float64) + patch_size_lv0_medium / 2.0
    dists = np.sum((centres - np.array([cx, cy])) ** 2, axis=1)
    return int(np.argmin(dists))


def iter_tissue_patches(
    svs_path: Path,
    *,
    objective: str = "20x",
    patch_size: int = 512,
    stride: int | None = None,
    background_threshold: int = 220,
    max_patches: int = 0,
    accept_patch: Callable[[np.ndarray, int, int, int], bool] | None = None,
) -> Iterator[tuple[object, tuple[int, int], int]]:
    """Yield (PIL RGB patch, level-0 coord, patch_size_lv0) for tissue tiles.

    Coordinates are top-left corners in level-0 pixel space, as required by TITAN.
    patch_size_lv0 is the spacing between adjacent patch origins at level 0.

    When *accept_patch* is None, falls back to grayscale mean <= *background_threshold*.
    Tiling should pass *max_patches=0* so all tissue coords are saved; subsampling
    happens at encode time via ``vision.patch_sampling.select_coords_for_encode``.
    """
    import openslide

    stride = stride or patch_size

    def _default_accept(arr: np.ndarray, _x0: int, _y0: int, _ps: int) -> bool:
        return is_tissue_patch(arr, background_threshold)

    predicate = accept_patch or _default_accept

    slide = openslide.OpenSlide(str(svs_path))
    try:
        level, level_downsample, patch_size_lv0 = _read_level_for_objective(
            slide, objective, patch_size
        )

        w, h = slide.level_dimensions[level]
        count = 0
        for y in range(0, max(h - patch_size + 1, 1), stride):
            for x in range(0, max(w - patch_size + 1, 1), stride):
                region = slide.read_region(
                    (int(x * level_downsample), int(y * level_downsample)),
                    level,
                    (patch_size, patch_size),
                ).convert("RGB")
                arr = np.asarray(region)
                x0 = int(x * level_downsample)
                y0 = int(y * level_downsample)
                if not predicate(arr, x0, y0, patch_size_lv0):
                    continue
                yield region, (x0, y0), patch_size_lv0
                count += 1
                if max_patches > 0 and count >= max_patches:
                    return
    finally:
        slide.close()
