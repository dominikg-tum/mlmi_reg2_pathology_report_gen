"""Config-driven magnification bands (variable native patch size, 224px CONCH input)."""

from __future__ import annotations

from vision.mag_config import encode_levels, mag_band_config, retrieval_config, top_k_for_zoom, zoom_config


def test_zoom_bands_native_patch_sizes():
    assert zoom_config("5x") == ("5x", 2048, 224)
    assert zoom_config("10x") == ("10x", 1024, 224)
    assert zoom_config("20x") == ("20x", 512, 224)
    assert zoom_config("40x") == ("40x", 256, 224)


def test_mag_band_config_objective_and_patch():
    assert mag_band_config("20x") == ("20x", 512)


def test_retrieval_encode_levels():
    assert encode_levels() == ["5x", "10x", "20x", "40x"]
    rcfg = retrieval_config()
    assert rcfg["kmeans_k"] == 100
    assert top_k_for_zoom("5x") == 3
    assert top_k_for_zoom("20x") == 5
    assert rcfg["d_min_20x_px"] == 512
