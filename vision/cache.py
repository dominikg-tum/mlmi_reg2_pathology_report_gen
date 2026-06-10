"""Paths to offline WSI caches (thumbnails + per-zoom patch embeddings)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vision.mag_config import VALID_ZOOM_LEVELS, normalize_zoom


@dataclass
class SlideCache:
    """Per-slide artifacts produced by offline scripts (never built at inference)."""

    slide_id: str
    cache_dir: Path | None = None
    thumbnail_path: Path | None = None
    slide_embedding_path: Path | None = None
    evidence_dir: Path | None = None

    def _path(self, zoom: str, kind: str) -> Path | None:
        if self.cache_dir is None:
            return None
        zoom = normalize_zoom(zoom)
        return self.cache_dir / f"{kind}_{zoom}.pt"

    def patch_embeddings_path(self, zoom: str) -> Path | None:
        """Primary artifact: patch_embeddings_{zoom}.pt (N × 768)."""
        zoom = normalize_zoom(zoom)
        if self.cache_dir is None:
            return None
        primary = self.cache_dir / f"patch_embeddings_{zoom}.pt"
        if primary.exists():
            return primary
        # Legacy band filenames (pre-zoom rename).
        legacy = {
            "5x": "embeddings_low.pt",
            "10x": "embeddings_medium.pt",
            "20x": "embeddings_high.pt",
            "40x": "embeddings_ultra.pt",
        }
        old = self.cache_dir / legacy.get(zoom, f"embeddings_{zoom}.pt")
        return old if old.exists() else primary

    def embedding_path_for_level(self, level: str) -> Path | None:
        """Retrieval API — level is a zoom key (5x/10x/20x/40x) or legacy band alias."""
        return self.patch_embeddings_path(level)

    def coords_path_for_level(self, level: str) -> Path | None:
        zoom = normalize_zoom(level)
        if self.cache_dir is None:
            return None
        primary = self.cache_dir / f"coords_{zoom}.pt"
        if primary.exists():
            return primary
        legacy = {
            "5x": "coords_low.pt",
            "10x": "coords_medium.pt",
            "20x": "coords_high.pt",
            "40x": "coords_ultra.pt",
        }
        old = self.cache_dir / legacy.get(zoom, f"coords_{level}.pt")
        return old if old.exists() else primary

    def centroid_path_for_level(self, level: str) -> Path | None:
        zoom = normalize_zoom(level)
        if self.cache_dir is None:
            return None
        primary = self.cache_dir / f"kmeans_centroids_{zoom}.pt"
        if primary.exists():
            return primary
        legacy = {
            "5x": "kmeans_centroids_low.pt",
            "10x": "kmeans_centroids_medium.pt",
            "20x": "kmeans_centroids_high.pt",
            "40x": "kmeans_centroids_ultra.pt",
        }
        old = self.cache_dir / legacy.get(zoom, f"kmeans_centroids_{level}.pt")
        return old if old.exists() else primary

    def meta_path_for_level(self, level: str) -> Path | None:
        if self.cache_dir is None:
            return None
        zoom = normalize_zoom(level)
        return self.cache_dir / f"meta_{zoom}.json"

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
    )
