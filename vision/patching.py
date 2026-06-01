"""Offline patch extraction (not called during agent inference)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def extract_patches(
    svs_path: Path,
    *,
    mag_band: str = "high",
    patch_size: int = 256,
    background_threshold: int = 220,
) -> tuple[list[Any], list[tuple[int, int]]]:
    """DOMI: implement with openslide — tile WSI, filter background, return patches + coords."""
    raise NotImplementedError(
        "Implement patch extraction in scripts/vision/encode_patches_offline.py"
    )
