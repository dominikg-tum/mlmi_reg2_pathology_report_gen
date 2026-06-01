"""Patch encoder protocol for offline encoding jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class PatchEncoder(Protocol):
    def encode_patches(self, patch_images: list) -> np.ndarray:
        """Return [N, D] embeddings."""
        ...


def encode_slide_offline(
    encoder: PatchEncoder,
    svs_path: Path,
    output_path: Path,
    *,
    mag_band: str = "high",
) -> None:
    """DOMI: extract patches, encode, save embeddings.pt + coords.pt."""
    raise NotImplementedError("Use scripts/vision/encode_patches_offline.py")
