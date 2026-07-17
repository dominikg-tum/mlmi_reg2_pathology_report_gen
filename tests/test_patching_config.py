"""Config-driven magnification bands (variable native patch size, 224px CONCH input)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from vision.mag_config import (
    encode_levels,
    fixed_retrieval_pool,
    include_grandparent,
    mag_band_config,
    parent_zoom_for,
    retrieval_config,
    tiling_encode_config,
    tissue_filter_config,
    top_k_for_zoom,
    zoom_config,
)
from vision import patching


def test_zoom_bands_native_patch_sizes():
    assert zoom_config("5x") == ("5x", 2048, 224)
    assert zoom_config("10x") == ("10x", 1024, 224)
    assert zoom_config("20x") == ("20x", 512, 224)
    assert zoom_config("40x") == ("40x", 256, 224)


def test_mag_band_config_objective_and_patch():
    assert mag_band_config("20x") == ("20x", 512)


def test_retrieval_encode_levels():
    assert encode_levels() == ["20x"]
    rcfg = retrieval_config()
    assert rcfg["kmeans_k"] == 100
    assert rcfg["search_all_patches"] is True
    assert fixed_retrieval_pool() == "20x"
    assert top_k_for_zoom("5x") == 3
    assert top_k_for_zoom("20x") == 5
    assert rcfg["d_min_20x_px"] == 512


def test_adjacent_scale_parent_map():
    assert parent_zoom_for("40x") is None
    assert parent_zoom_for("20x") is None
    assert parent_zoom_for("10x") is None
    assert parent_zoom_for("5x") is None


def test_include_grandparent_for_integration_nodes():
    assert include_grandparent(tier="integration")
    assert include_grandparent(node_kind="report")
    assert not include_grandparent(tier="local_features", node_kind="local")


def test_tiling_encode_policy_defaults():
    tcfg = tiling_encode_config()
    assert tcfg["max_patches_per_slide"] == 4096
    assert tcfg["full_encode_threshold"] == 1024
    assert tcfg["grid_cells"] == (8, 8)


def test_tissue_filter_slide_mask_default():
    fcfg = tissue_filter_config()
    assert fcfg["method"] == "slide_mask"
    assert fcfg["min_tissue_fraction"] == 0.40


def test_background_threshold_override_reaches_accept_fn():
    """Caller override must drive mean_threshold; None keeps YAML default."""
    yaml_val = int(tissue_filter_config().get("background_threshold", 220))
    assert patching._resolve_background_threshold(None) == yaml_val
    assert patching._resolve_background_threshold(180) == 180

    fake_cfg = {
        **tissue_filter_config(),
        "method": "mean_threshold",
        "background_threshold": 220,
    }
    white = np.full((8, 8, 3), 200, dtype=np.uint8)
    with patch.object(patching, "tissue_filter_config", return_value=fake_cfg):
        fn_yaml = patching._patch_accept_fn_for_slide(Path("dummy.svs"))
        assert fn_yaml(white, 0, 0, 8)  # 200 <= 220
        fn_strict = patching._patch_accept_fn_for_slide(
            Path("dummy.svs"), background_threshold=100
        )
        assert not fn_strict(white, 0, 0, 8)  # 200 > 100
