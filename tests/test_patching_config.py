"""Config-driven magnification bands (512px locked)."""

from __future__ import annotations

from vision.mag_config import encode_levels, mag_band_config, retrieval_config


def test_mag_bands_512px_and_objectives():
    assert mag_band_config("low") == ("4x", 512)
    assert mag_band_config("medium") == ("10x", 512)
    assert mag_band_config("high") == ("20x", 512)


def test_retrieval_encode_levels():
    assert encode_levels() == ["medium", "high"]
    rcfg = retrieval_config()
    assert rcfg["kmeans_k"] == 100
    assert rcfg["top_k"] == 5
    assert rcfg["d_min_20x_px"] == 512
