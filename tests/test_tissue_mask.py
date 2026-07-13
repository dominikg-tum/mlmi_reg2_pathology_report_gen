"""Slide-level HSV tissue mask helpers."""

from __future__ import annotations

import numpy as np

from vision.tissue_mask import (
    SlideTissueMask,
    build_hsv_tissue_mask,
    make_patch_accept_fn,
)


def test_hsv_mask_keeps_pink_tissue_drops_white_glass():
    tissue = np.zeros((64, 64, 3), dtype=np.uint8)
    tissue[:, :, 0] = 180
    tissue[:, :, 1] = 80
    tissue[:, :, 2] = 120
    glass = np.full((64, 64, 3), 245, dtype=np.uint8)
    rgb = np.vstack([tissue, glass])
    mask = build_hsv_tissue_mask(rgb, sat_min=0.08, val_max=0.95, morph_close_px=0)
    assert mask[:64].mean() > 200
    assert mask[64:].mean() < 50


def test_slide_mask_tissue_fraction():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:60, 10:60] = 255
    slide_mask = SlideTissueMask(mask=mask, level0_width=1000, level0_height=1000)
    full = slide_mask.tissue_fraction(0, 0, patch_size_lv0=500)
    empty = slide_mask.tissue_fraction(700, 700, patch_size_lv0=200)
    assert full > 0.2
    assert empty == 0.0


def test_make_patch_accept_fn_mean_threshold():
    accept = make_patch_accept_fn(method="mean_threshold", background_threshold=220)
    white = np.full((8, 8, 3), 240, dtype=np.uint8)
    tissue = np.full((8, 8, 3), 120, dtype=np.uint8)
    assert not accept(white, 0, 0, 512)
    assert accept(tissue, 0, 0, 512)
