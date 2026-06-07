"""Paths to offline WSI caches (thumbnails + per-mag patch embeddings)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SlideCache:
    """Per-slide artifacts produced by offline scripts (never built at inference)."""

    slide_id: str
    cache_dir: Path | None = None
    thumbnail_path: Path | None = None
    slide_embedding_path: Path | None = None
    evidence_dir: Path | None = None
    embeddings_low: Path | None = None
    embeddings_medium: Path | None = None
    embeddings_high: Path | None = None
    coords_low: Path | None = None
    coords_medium: Path | None = None
    coords_high: Path | None = None
    kmeans_centroids_low: Path | None = None
    kmeans_centroids_medium: Path | None = None
    kmeans_centroids_high: Path | None = None

    def embedding_path_for_level(self, level: str) -> Path | None:
        return {
            "low": self.embeddings_low,
            "medium": self.embeddings_medium,
            "high": self.embeddings_high,
        }.get(level)

    def coords_path_for_level(self, level: str) -> Path | None:
        return {
            "low": self.coords_low,
            "medium": self.coords_medium,
            "high": self.coords_high,
        }.get(level)

    def centroid_path_for_level(self, level: str) -> Path | None:
        return {
            "low": self.kmeans_centroids_low,
            "medium": self.kmeans_centroids_medium,
            "high": self.kmeans_centroids_high,
        }.get(level)

    def meta_path_for_level(self, level: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"meta_{level}.json"

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


def build_slide_cache(cache_root: Path, slide_id: str) -> SlideCache:
    d = slide_cache_dir(cache_root, slide_id)
    thumb = d / "thumbnail.png"
    return SlideCache(
        slide_id=slide_id,
        cache_dir=d,
        thumbnail_path=thumb if thumb.exists() else None,
        slide_embedding_path=d / "slide_embedding.pt",
        evidence_dir=d / "evidence",
        embeddings_low=d / "embeddings_low.pt",
        embeddings_medium=d / "embeddings_medium.pt",
        embeddings_high=d / "embeddings_high.pt",
        coords_low=d / "coords_low.pt",
        coords_medium=d / "coords_medium.pt",
        coords_high=d / "coords_high.pt",
        kmeans_centroids_low=d / "kmeans_centroids_low.pt",
        kmeans_centroids_medium=d / "kmeans_centroids_medium.pt",
        kmeans_centroids_high=d / "kmeans_centroids_high.pt",
    )
