"""Offline patch extraction (not called during agent inference)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vision.mag_config import mag_band_config
from vision.wsi_io import iter_tissue_patches, load_patch_at_coord


def extract_patch_coords(
    svs_path: Path,
    *,
    mag_band: str = "20x",
    background_threshold: int = 220,
    max_patches: int = 0,
    stride: int | None = None,
) -> tuple[list[tuple[int, int]], int]:
    """Tile WSI and return level-0 coords only (no patch images retained)."""
    objective, patch_size = mag_band_config(mag_band)
    coords: list[tuple[int, int]] = []
    patch_size_lv0 = patch_size

    for _, (x0, y0), ps_lv0 in iter_tissue_patches(
        svs_path,
        objective=objective,
        patch_size=patch_size,
        stride=stride,
        background_threshold=background_threshold,
        max_patches=max_patches,
    ):
        coords.append((x0, y0))
        patch_size_lv0 = ps_lv0

    return coords, patch_size_lv0


def extract_patches(
    svs_path: Path,
    *,
    mag_band: str = "20x",
    patch_size: int | None = None,
    background_threshold: int = 220,
    max_patches: int = 0,
    stride: int | None = None,
) -> tuple[list[Any], list[tuple[int, int]], int]:
    """Tile WSI, filter background, return PIL patches + level-0 coords + patch_size_lv0."""
    objective, cfg_patch_size = mag_band_config(mag_band)
    patch_size = patch_size or cfg_patch_size
    patches: list[Any] = []
    coords: list[tuple[int, int]] = []
    patch_size_lv0 = patch_size

    for img, (x0, y0), ps_lv0 in iter_tissue_patches(
        svs_path,
        objective=objective,
        patch_size=patch_size,
        stride=stride,
        background_threshold=background_threshold,
        max_patches=max_patches,
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
