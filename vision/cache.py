"""Paths to offline WSI caches (thumbnails + per-zoom patch embeddings)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vision.mag_config import (
    VALID_THUMBNAIL_VARIANTS,
    VALID_ZOOM_LEVELS,
    load_vision_config,
    normalize_zoom,
    thumbnail_config,
)
from vision.wsi_mapping import canonical_slide_id


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
    canonical = canonical_slide_id(slide_id)
    safe = canonical.replace(",", "_").replace("/", "_")
    return cache_root / safe


def slide_id_to_stem(slide_id: str) -> str:
    """TUM_Uterus_0001.svs → TUM_Uterus_0001 (matches dataset JPEG stems)."""
    canonical = canonical_slide_id(slide_id)
    if canonical.lower().endswith(".svs"):
        return canonical[:-4]
    return canonical


def dataset_thumbnail_dir(vcfg: dict | None = None) -> Path | None:
    """Configured team bank directory, or None if dataset thumbnails are disabled."""
    tcfg = thumbnail_config() if vcfg is None else vcfg.get("thumbnail", {})
    root = str(tcfg.get("dataset_root", "")).strip()
    variant = str(tcfg.get("variant", "thumbnails")).strip()
    if not root:
        return None
    if variant not in VALID_THUMBNAIL_VARIANTS:
        raise ValueError(
            f"thumbnail.variant must be one of {VALID_THUMBNAIL_VARIANTS}; got {variant!r}"
        )
    return Path(root).expanduser() / variant


def dataset_thumbnail_path(slide_id: str, vcfg: dict | None = None) -> Path | None:
    """Path under dataset/{variant}/ for this slide, if the file exists."""
    bank = dataset_thumbnail_dir(vcfg)
    if bank is None:
        return None
    stem = slide_id_to_stem(slide_id)
    for ext in (".jpg", ".jpeg", ".png"):
        path = bank / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def cache_thumbnail_path(cache_root: Path, slide_id: str) -> Path | None:
    """Per-slide offline thumbnail under cache_root/{slide_id}/."""
    d = slide_cache_dir(cache_root, slide_id)
    for name in ("thumbnail.png", "thumbnail.jpg", "thumbnail.jpeg"):
        path = d / name
        if path.exists():
            return path
    return None


def resolve_thumbnail_path(
    cache_root: Path,
    slide_id: str,
    *,
    vcfg: dict | None = None,
) -> Path | None:
    """Dataset bank first (configs/vision.yaml), then per-slide cache_root artifact."""
    vcfg = vcfg or load_vision_config()
    return dataset_thumbnail_path(slide_id, vcfg) or cache_thumbnail_path(
        cache_root, slide_id
    )


def build_slide_cache(
    cache_root: Path,
    slide_id: str,
    *,
    vcfg: dict | None = None,
) -> SlideCache:
    # Store the canonical id: caches, the thumbnail bank and the raw .svs on disk
    # are all keyed by TUM slide_id, while chains carry disk_name UUIDs.
    canonical = canonical_slide_id(slide_id)
    d = slide_cache_dir(cache_root, canonical)
    thumb = resolve_thumbnail_path(cache_root, canonical, vcfg=vcfg)
    return SlideCache(
        slide_id=canonical,
        cache_dir=d,
        thumbnail_path=thumb,
        slide_embedding_path=d / "slide_embedding.pt",
        evidence_dir=d / "evidence",
    )
