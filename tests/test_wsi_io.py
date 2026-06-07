"""Tests for vision/wsi_io helpers (no openslide required)."""

from __future__ import annotations

import numpy as np

import numpy as np

from vision.wsi_io import find_parent_patch_index, is_tissue_patch, slide_id_from_path


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
