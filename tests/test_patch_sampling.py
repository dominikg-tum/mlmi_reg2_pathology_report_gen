"""Stratified patch coordinate selection."""

from __future__ import annotations

from vision.encode_selection import coords_for_encode
from vision.patch_sampling import select_coords_for_encode, stratified_grid_sample


def test_select_coords_full_when_under_threshold():
    coords = [(i * 100, 0) for i in range(50)]
    selected, mode, meta = coords_for_encode(
        coords,
        patch_size_lv0=512,
        full_encode_threshold=1024,
        max_patches=4096,
    )
    assert mode == "full"
    assert len(selected) == 50
    assert meta["n_patches_tiled"] == 50
    assert meta["n_patches_encoded"] == 50


def test_select_coords_stratified_when_above_threshold():
    coords = [(i * 10, (i % 8) * 500) for i in range(2000)]
    selected, mode, meta = coords_for_encode(
        coords,
        patch_size_lv0=512,
        full_encode_threshold=1024,
        max_patches=512,
        grid_cells=(4, 4),
    )
    assert mode == "stratified"
    assert len(selected) == 512
    assert meta["n_patches_tiled"] == 2000


def test_stratified_spreads_across_grid():
    # Two spatial clusters — stratified should pick from both
    left = [(100 + i * 20, 100) for i in range(50)]
    right = [(9000 + i * 20, 9000) for i in range(50)]
    coords = left + right
    picked = stratified_grid_sample(coords, k=8, grid_cells=(2, 2), patch_size_lv0=512)
    xs = [c[0] for c in picked]
    assert any(x < 5000 for x in xs)
    assert any(x >= 5000 for x in xs)


def test_select_coords_direct_api():
    coords = [(0, 0), (1000, 1000)]
    out, mode = select_coords_for_encode(
        coords, full_encode_threshold=10, max_patches=4096, grid_cells=(2, 2)
    )
    assert mode == "full"
    assert len(out) == 2


def test_stratified_handles_zero_quota_cells():
    # More occupied cells than k => some cells get quota 0; must not divide by zero.
    coords = [(x * 1000, y * 1000) for y in range(4) for x in range(4)]
    picked = stratified_grid_sample(coords, k=3, grid_cells=(4, 4), patch_size_lv0=512)
    assert len(picked) == 3
    assert len(set(picked)) == 3
