"""Offline patch extraction (not called during agent inference)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vision.wsi_io import iter_tissue_patches


def extract_patches(
    svs_path: Path,
    *,
    mag_band: str = "high",
    patch_size: int = 256,
    background_threshold: int = 220,
    max_patches: int = 0,
    stride: int | None = None,
) -> tuple[list[Any], list[tuple[int, int]], int]:
    """Tile WSI, filter background, return PIL patches + level-0 coords + patch_size_lv0."""
    objective = {"low": "5x", "medium": "10x", "high": "20x"}.get(mag_band, "20x")
    patches: list[Any] = []
    coords: list[tuple[int, int]] = []
    patch_size_lv0 = 512

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
