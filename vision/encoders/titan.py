"""TITAN offline encoder stub — DOMI implements against cluster repos/TITAN."""

from __future__ import annotations

import numpy as np


class TitanPatchEncoder:
    def __init__(self, model_dir=None):
        self.model_dir = model_dir

    def encode_patches(self, patch_images: list) -> np.ndarray:
        raise NotImplementedError(
            "Load TITAN image encoder from cluster repos/TITAN; "
            "run only inside scripts/vision/encode_patches_offline.py"
        )
