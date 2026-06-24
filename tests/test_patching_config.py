"""Config-driven magnification bands (variable native patch size, 224px CONCH input)."""

from __future__ import annotations

from vision.mag_config import (
    encode_levels,
    include_grandparent,
    mag_band_config,
    parent_zoom_for,
    retrieval_config,
    top_k_for_zoom,
    zoom_config,
)


def test_zoom_bands_native_patch_sizes():
    assert zoom_config("5x") == ("5x", 2048, 224)
    assert zoom_config("10x") == ("10x", 1024, 224)
    assert zoom_config("20x") == ("20x", 512, 224)
    assert zoom_config("40x") == ("40x", 256, 224)


def test_mag_band_config_objective_and_patch():
    assert mag_band_config("20x") == ("20x", 512)


def test_retrieval_encode_levels():
    assert encode_levels() == ["5x", "10x", "20x"]
    rcfg = retrieval_config()
    assert rcfg["kmeans_k"] == 100
    assert top_k_for_zoom("5x") == 3
    assert top_k_for_zoom("20x") == 5
    assert rcfg["d_min_20x_px"] == 512


def test_uni2_config_levels():
    from vision.mag_config import load_vision_config

    cfg = load_vision_config()["uni2"]
    assert cfg["levels"] == ["1.25x", "2.5x", "5x", "10x"]
    assert cfg["patch_size"] == 224


def test_adjacent_scale_parent_map():
    assert parent_zoom_for("40x") == "20x"
    assert parent_zoom_for("20x") == "10x"
    assert parent_zoom_for("10x") == "5x"
    assert parent_zoom_for("5x") is None


def test_include_grandparent_for_integration_nodes():
    assert include_grandparent(tier="integration")
    assert include_grandparent(node_kind="report")
    assert not include_grandparent(tier="local_features", node_kind="local")
