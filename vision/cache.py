"""Paths to offline WSI caches (thumbnails + per-mag patch embeddings)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SlideCache:
    """Per-slide artifacts produced by offline scripts (never built at inference)."""

    slide_id: str
    thumbnail_path: Path | None = None
    slide_embedding_path: Path | None = None
    evidence_dir: Path | None = None
    embeddings_low: Path | None = None
    embeddings_mid: Path | None = None
    embeddings_high: Path | None = None
    coords_low: Path | None = None
    coords_mid: Path | None = None
    coords_high: Path | None = None

    def embedding_path_for_level(self, level: str) -> Path | None:
        return {
            "low": self.embeddings_low,
            "medium": self.embeddings_mid,
            "high": self.embeddings_high,
        }.get(level)

    def load_slide_embedding(self):
        """Load offline TITAN slide vector [D] or None."""
        path = self.slide_embedding_path
        if path is None or not path.exists():
            return None
        import torch

        data = torch.load(path, map_location="cpu", weights_only=False)
        if hasattr(data, "numpy"):
            return data.numpy().reshape(-1)
        return np.asarray(data, dtype=np.float32).reshape(-1)

    def evidence_patch_paths(self) -> list[Path]:
        if self.evidence_dir is None or not self.evidence_dir.is_dir():
            return []
        return sorted(self.evidence_dir.glob("*.png"))


def slide_cache_dir(cache_root: Path, slide_id: str) -> Path:
    safe = slide_id.replace(",", "_").replace("/", "_")
    return cache_root / safe
