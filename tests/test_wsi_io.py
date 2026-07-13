"""Tests for vision/wsi_io helpers (no openslide required)."""

from __future__ import annotations

import numpy as np

import numpy as np

from vision.wsi_io import _zoom_crop_topleft, find_parent_patch_index, is_tissue_patch, slide_id_from_path


def test_slide_id_from_path():
    assert slide_id_from_path(__import__("pathlib").Path("a/b/case01.svs")) == "case01.svs"


def test_is_tissue_patch_filters_white():
    white = np.full((512, 512, 3), 240, dtype=np.uint8)
    tissue = np.full((512, 512, 3), 120, dtype=np.uint8)
    assert not is_tissue_patch(white, background_threshold=220)
    assert is_tissue_patch(tissue, background_threshold=220)


def test_find_parent_patch_index_contains_centre():
    coords_medium = np.array([[0, 0], [1000, 0], [0, 1000]], dtype=np.int64)
    idx = find_parent_patch_index(
        (120, 120),
        coords_medium,
        patch_size_lv0_high=200,
        patch_size_lv0_medium=1000,
    )
    assert idx == 0


def test_zoom_crop_topleft_recentres_to_target_patch():
    # from: 512px tile at level-0, to: 256px tile (40x-like)
    # center of from tile at (1000+256, 2000+256) => (1256, 2256)
    # topleft should be (1256-128, 2256-128) => (1128, 2128)
    x1, y1 = _zoom_crop_topleft(
        (1000, 2000),
        from_patch_size_lv0=512,
        to_patch_size_lv0=256,
    )
    assert (x1, y1) == (1128, 2128)
