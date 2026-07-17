"""Offline patch extraction (not called during agent inference)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vision.mag_config import mag_band_config, tissue_filter_config, thumbnail_config
from vision.tissue_mask import build_mask_from_thumbnail, make_patch_accept_fn
from vision.wsi_io import iter_tissue_patches, load_patch_at_coord


def _resolve_background_threshold(background_threshold: int | None) -> int:
    if background_threshold is not None:
        return int(background_threshold)
    return int(tissue_filter_config().get("background_threshold", 220))


def _patch_accept_fn_for_slide(
    svs_path: Path,
    *,
    background_threshold: int | None = None,
):
    """Build tissue gate from configs/vision.yaml (slide_mask default).

    When *background_threshold* is None, use tissue_filter.background_threshold
    from YAML (default 220). Caller override applies to mean_threshold method.
    """
    tcfg = tissue_filter_config()
    method = str(tcfg.get("method", "slide_mask"))
    bg_thresh = _resolve_background_threshold(background_threshold)
    if method == "slide_mask":
        thumb_cfg = thumbnail_config()
        mask = build_mask_from_thumbnail(
            svs_path,
            max_edge_px=int(thumb_cfg.get("max_edge_px", 1024)),
            sat_min=float(tcfg.get("hsv_sat_min", 0.08)),
            val_max=float(tcfg.get("hsv_val_max", 0.95)),
            morph_close_px=int(tcfg.get("morph_close_px", 5)),
        )
        return make_patch_accept_fn(
            method=method,
            min_tissue_fraction=float(tcfg.get("min_tissue_fraction", 0.40)),
            tissue_mask=mask,
            background_threshold=bg_thresh,
        )
    return make_patch_accept_fn(
        method=method,
        background_threshold=bg_thresh,
    )


def extract_patch_coords(
    svs_path: Path,
    *,
    mag_band: str = "20x",
    background_threshold: int | None = None,
    max_patches: int = 0,
    stride: int | None = None,
) -> tuple[list[tuple[int, int]], int]:
    """Tile WSI and return level-0 coords only (no patch images retained).

    *background_threshold* overrides YAML tissue_filter.background_threshold for
    mean_threshold filtering; None keeps the YAML default.
    """
    objective, patch_size = mag_band_config(mag_band)
    coords: list[tuple[int, int]] = []
    patch_size_lv0 = patch_size

    bg_thresh = _resolve_background_threshold(background_threshold)
    accept_patch = _patch_accept_fn_for_slide(
        svs_path, background_threshold=bg_thresh
    )
    for _, (x0, y0), ps_lv0 in iter_tissue_patches(
        svs_path,
        objective=objective,
        patch_size=patch_size,
        stride=stride,
        background_threshold=bg_thresh,
        max_patches=max_patches,
        accept_patch=accept_patch,
    ):
        coords.append((x0, y0))
        patch_size_lv0 = ps_lv0

    return coords, patch_size_lv0


def extract_patches(
    svs_path: Path,
    *,
    mag_band: str = "20x",
    patch_size: int | None = None,
    background_threshold: int | None = None,
    max_patches: int = 0,
    stride: int | None = None,
) -> tuple[list[Any], list[tuple[int, int]], int]:
    """Tile WSI, filter background, return PIL patches + level-0 coords + patch_size_lv0.

    *background_threshold* overrides YAML tissue_filter.background_threshold for
    mean_threshold filtering; None keeps the YAML default.
    """
    objective, cfg_patch_size = mag_band_config(mag_band)
    patch_size = patch_size or cfg_patch_size
    patches: list[Any] = []
    coords: list[tuple[int, int]] = []
    patch_size_lv0 = patch_size

    bg_thresh = _resolve_background_threshold(background_threshold)
    accept_patch = _patch_accept_fn_for_slide(
        svs_path, background_threshold=bg_thresh
    )
    for img, (x0, y0), ps_lv0 in iter_tissue_patches(
        svs_path,
        objective=objective,
        patch_size=patch_size,
        stride=stride,
        background_threshold=bg_thresh,
        max_patches=max_patches,
        accept_patch=accept_patch,
    ):
        patches.append(img)
        coords.append((x0, y0))
        patch_size_lv0 = ps_lv0

    return patches, coords, patch_size_lv0


def load_patches_from_coords(
    svs_path: Path,
    coords: list[tuple[int, int]],
    *,
    mag_band: str = "20x",
    patch_size: int | None = None,
) -> list[Any]:
    """Re-read tissue patches from WSI at precomputed level-0 coordinates."""
    objective, cfg_patch_size = mag_band_config(mag_band)
    patch_size = patch_size or cfg_patch_size
    return [
        load_patch_at_coord(
            svs_path,
            coord,
            objective=objective,
            patch_size=patch_size,
        )
        for coord in coords
    ]
