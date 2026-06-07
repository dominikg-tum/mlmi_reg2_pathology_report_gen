"""Magnification band config from configs/vision.yaml (single source of truth)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
VISION_CONFIG_PATH = REPO_ROOT / "configs" / "vision.yaml"

VALID_MAG_BANDS = ("low", "medium", "high")


@lru_cache(maxsize=1)
def load_vision_config() -> dict:
    with VISION_CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def mag_band_config(mag_band: str) -> tuple[str, int]:
    """Return (objective, patch_size) for a band key."""
    if mag_band not in VALID_MAG_BANDS:
        raise ValueError(f"Unknown mag_band {mag_band!r}; use one of {VALID_MAG_BANDS}")
    bands = load_vision_config().get("magnification_bands", {})
    band = bands.get(mag_band, {})
    objective = str(band.get("objective", "20x"))
    patch_size = int(band.get("patch_size", 512))
    return objective, patch_size


def retrieval_config() -> dict:
    return load_vision_config().get("retrieval", {})


def encode_levels() -> list[str]:
    levels = retrieval_config().get("encode_levels", ["medium", "high"])
    return list(levels)
