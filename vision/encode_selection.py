"""Shared offline encode coordinate selection."""

from __future__ import annotations

from vision.mag_config import tiling_encode_config
from vision.patch_sampling import select_coords_for_encode


def coords_for_encode(
    coords: list[tuple[int, int]],
    *,
    patch_size_lv0: int = 0,
    max_patches: int | None = None,
    full_encode_threshold: int | None = None,
    grid_cells: tuple[int, int] | None = None,
) -> tuple[list[tuple[int, int]], str, dict]:
    """Apply tiling encode policy from configs/vision.yaml."""
    cfg = tiling_encode_config()
    cap = int(max_patches) if max_patches is not None else int(cfg["max_patches_per_slide"])
    threshold = (
        int(full_encode_threshold)
        if full_encode_threshold is not None
        else int(cfg["full_encode_threshold"])
    )
    cells = grid_cells if grid_cells is not None else tuple(cfg["grid_cells"])

    selected, mode = select_coords_for_encode(
        coords,
        full_encode_threshold=threshold,
        max_patches=cap,
        grid_cells=cells,
        patch_size_lv0=patch_size_lv0,
    )
    meta = {
        "n_patches_tiled": len(coords),
        "n_patches_encoded": len(selected),
        "sampling_mode": mode,
        "full_encode_threshold": threshold,
        "max_patches_per_slide": cap,
        "grid_cells": list(cells),
    }
    return selected, mode, meta
