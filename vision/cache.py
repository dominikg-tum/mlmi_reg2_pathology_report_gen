"""Paths to offline WSI caches (thumbnails + per-mag patch embeddings)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SlideCache:
    """Per-slide artifacts produced by offline scripts (never built at inference)."""

    slide_id: str
    thumbnail_path: Path | None = None
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


def slide_cache_dir(cache_root: Path, slide_id: str) -> Path:
    safe = slide_id.replace(",", "_").replace("/", "_")
    return cache_root / safe
