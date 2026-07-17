"""Spatially balanced patch coordinate selection for offline CONCH encode."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def stratified_grid_sample(
    coords: Sequence[tuple[int, int]],
    *,
    k: int,
    grid_cells: tuple[int, int] = (8, 8),
    patch_size_lv0: int = 0,
) -> list[tuple[int, int]]:
    """Pick up to *k* coords with roughly equal coverage over an NxM spatial grid.

    Coordinates are top-left corners in level-0 pixel space. When *k* >= len(coords),
    every coordinate is returned (order may differ from input).
    """
    if k <= 0 or not coords:
        return []
    pts = list(coords)
    if len(pts) <= k:
        return pts

    rows, cols = int(grid_cells[0]), int(grid_cells[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"grid_cells must be positive, got {grid_cells!r}")

    arr = np.asarray(pts, dtype=np.int64)
    half = max(int(patch_size_lv0) // 2, 1)
    centres = arr.astype(np.float64) + half

    x_min, y_min = centres.min(axis=0)
    x_max, y_max = centres.max(axis=0)
    span_x = max(float(x_max - x_min), 1.0)
    span_y = max(float(y_max - y_min), 1.0)

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (cx, cy) in enumerate(centres):
        gx = min(int((cx - x_min) / span_x * cols), cols - 1)
        gy = min(int((cy - y_min) / span_y * rows), rows - 1)
        buckets[(gy, gx)].append(i)

    cell_keys = sorted(buckets.keys())
    n_cells = len(cell_keys)
    base = k // n_cells
    remainder = k % n_cells

    selected: list[int] = []
    for cell_i, key in enumerate(cell_keys):
        quota = base + (1 if cell_i < remainder else 0)
        members = buckets[key]
        if quota >= len(members):
            selected.extend(members)
        else:
            step = max(len(members) // quota, 1)
            picked = members[::step][:quota]
            selected.extend(picked)

    if len(selected) > k:
        step = max(len(selected) // k, 1)
        selected = selected[::step][:k]
    elif len(selected) < k:
        remaining = [i for i in range(len(pts)) if i not in set(selected)]
        need = k - len(selected)
        if remaining:
            extra_step = max(len(remaining) // need, 1)
            selected.extend(remaining[::extra_step][:need])

    return [pts[i] for i in selected[:k]]


def select_coords_for_encode(
    coords: Sequence[tuple[int, int]],
    *,
    full_encode_threshold: int,
    max_patches: int,
    grid_cells: tuple[int, int] = (8, 8),
    patch_size_lv0: int = 0,
) -> tuple[list[tuple[int, int]], str]:
    """Choose which tiled coords to CONCH-encode.

    Returns (selected_coords, sampling_mode) where sampling_mode is one of:
    - ``full`` — every tissue coord is encoded (n <= full_encode_threshold)
    - ``stratified`` — spatial grid subsample to min(n, max_patches)
    - ``empty`` — no coords
    """
    pts = list(coords)
    n = len(pts)
    if n == 0:
        return [], "empty"

    threshold = max(int(full_encode_threshold), 0)
    if threshold > 0 and n <= threshold:
        return pts, "full"

    cap = int(max_patches) if max_patches > 0 else n
    target = min(n, cap)
    if target >= n:
        return pts, "full"

    sampled = stratified_grid_sample(
        pts, k=target, grid_cells=grid_cells, patch_size_lv0=patch_size_lv0
    )
    return sampled, "stratified"
