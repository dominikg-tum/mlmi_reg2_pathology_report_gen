"""UNI2-h patch encoder adapter.

The upstream UNI repository is expected to be cloned locally, e.g. /Volumes/Xun/UNI.
This module keeps the dependency import lazy so ordinary tests and frontend runs do
not import timm/torchvision or download weights.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


class UNI2Encoder:
    """Encode PIL image patches with MahmoodLab UNI2-h."""

    def __init__(
        self,
        *,
        repo_path: Path | str,
        model_name: str = "uni2-h",
        device: str | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser()
        self.model_name = model_name
        self.device = device
        self._model = None
        self._transform = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.repo_path.exists():
            raise FileNotFoundError(f"UNI repo not found: {self.repo_path}")
        repo_str = str(self.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        import torch
        from uni.get_encoder.get_encoder import get_encoder

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        model, transform = get_encoder(
            enc_name=self.model_name,
            img_resize=224,
            center_crop=True,
            device=dev,
        )
        if model is None or transform is None:
            raise RuntimeError(f"Could not load UNI encoder {self.model_name!r}")
        self._model = model
        self._transform = transform

    def encode_patches(self, patch_images: list[Any], *, batch_size: int = 16) -> np.ndarray:
        self._ensure_loaded()
        import torch

        feats: list[np.ndarray] = []
        for start in range(0, len(patch_images), batch_size):
            batch = patch_images[start : start + batch_size]
            tensors = torch.stack([self._transform(img.convert("RGB")) for img in batch])
            tensors = tensors.to(self.device)
            with torch.inference_mode():
                out = self._model(tensors)
            feats.append(out.detach().float().cpu().numpy())
        if not feats:
            return np.zeros((0, 1536), dtype=np.float32)
        return np.vstack(feats).astype(np.float32)


def mean_pool_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Simple WSI embedding baseline: mean-pool patch embeddings."""
    if embeddings.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(embeddings, dtype=np.float32).mean(axis=0).astype(np.float32)

