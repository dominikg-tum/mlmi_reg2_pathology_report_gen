"""Magnification / zoom config from configs/vision.yaml (single source of truth)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
VISION_CONFIG_PATH = REPO_ROOT / "configs" / "vision.yaml"

VALID_ZOOM_LEVELS = ("5x", "10x", "20x", "40x")

# Runtime zoom menu for ReAct (pixels only, not offline CONCH pools).
RUNTIME_ZOOM_LEVELS = ("10x", "20x", "40x")

# Legacy aliases → canonical zoom keys used in graph JSON and artifact names.
_ZOOM_ALIASES = {
    "4x": "5x",
    "low": "5x",
    "medium": "10x",
    "high": "20x",
    "ultra": "40x",
}

# Deprecated — use VALID_ZOOM_LEVELS; kept for scripts importing VALID_MAG_BANDS.
VALID_MAG_BANDS = VALID_ZOOM_LEVELS


@lru_cache(maxsize=1)
def load_vision_config() -> dict:
    with VISION_CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def normalize_zoom(zoom: str) -> str:
    """Map legacy band / 4x keys to canonical 5x/10x/20x/40x."""
    key = zoom.lower().replace("×", "x").strip()
    return _ZOOM_ALIASES.get(key, key)


def zoom_config(zoom: str) -> tuple[str, int, int]:
    """Return (objective, native_patch_px, conch_input_px) for a zoom level."""
    zoom = normalize_zoom(zoom)
    if zoom not in VALID_ZOOM_LEVELS:
        raise ValueError(f"Unknown zoom {zoom!r}; use one of {VALID_ZOOM_LEVELS}")
    bands = load_vision_config().get("magnification_bands", {})
    band = bands.get(zoom, {})
    objective = str(band.get("objective", zoom))
    patch_size = int(band.get("patch_size", 512))
    conch_input_px = int(band.get("conch_input_px", 224))
    return objective, patch_size, conch_input_px


def mag_band_config(mag_band: str) -> tuple[str, int]:
    """Return (objective, native_patch_px). Accepts zoom key or legacy band alias."""
    objective, patch_size, _ = zoom_config(mag_band)
    return objective, patch_size


def zoom_to_mag_band(zoom_level: str) -> str:
    """Canonical zoom key for cache lookup (identity after normalization)."""
    return normalize_zoom(zoom_level)


def mag_band_to_zoom(mag_band: str) -> str:
    return normalize_zoom(mag_band)


def thumbnail_config() -> dict:
    return load_vision_config().get("thumbnail", {})


VALID_THUMBNAIL_VARIANTS = ("thumbnails", "thumbnails_kmeans", "thumbnails_kmeans_5")


def retrieval_config() -> dict:
    return load_vision_config().get("retrieval", {})


def default_search_all_patches() -> bool:
    """Whether Phase 1 cosine retrieval ranks the full offline pool (default: true)."""
    return bool(retrieval_config().get("search_all_patches", True))


def fixed_retrieval_pool() -> str:
    """CONCH retrieval pool zoom key (default: 20x)."""
    pool = str(retrieval_config().get("fixed_pool", "20x"))
    pool = normalize_zoom(pool)
    if pool not in VALID_ZOOM_LEVELS:
        raise ValueError(
            f"retrieval.fixed_pool must be one of {VALID_ZOOM_LEVELS}; got {pool!r}"
        )
    return pool


def clamp_runtime_zoom(zoom: str) -> str:
    """Clamp arbitrary zoom strings to the allowed runtime zoom menu."""
    key = normalize_zoom(zoom).replace("1.25x", "5x")
    if key in RUNTIME_ZOOM_LEVELS:
        return key
    if key in ("4x", "5x"):
        return "10x"
    return "20x"


def tissue_filter_config() -> dict:
    return load_vision_config().get("tissue_filter", {})


def titan_config() -> dict:
    return load_vision_config().get("titan", {})


def tiling_encode_config() -> dict:
    """Patch cap / stratified sampling knobs for offline CONCH encode."""
    tcfg = titan_config()
    return {
        "max_patches_per_slide": int(tcfg.get("max_patches_per_slide", 4096)),
        "full_encode_threshold": int(tcfg.get("full_encode_threshold", 1024)),
        "patch_sampling": str(tcfg.get("patch_sampling", "stratified_grid")),
        "grid_cells": tuple(int(x) for x in tcfg.get("grid_cells", [8, 8])),
    }


def encode_levels() -> list[str]:
    levels = retrieval_config().get("encode_levels", list(VALID_ZOOM_LEVELS))
    return [normalize_zoom(z) for z in levels]


def top_k_for_zoom(zoom: str) -> int:
    """k=3 for global (5×); k=5 for cellular (20×, 40×); 10× uses config default."""
    zoom = normalize_zoom(zoom)
    rcfg = retrieval_config()
    by_zoom = rcfg.get("top_k_by_zoom", {})
    if zoom in by_zoom:
        return int(by_zoom[zoom])
    return int(rcfg.get("top_k", 5))


_DEFAULT_PARENT_MAP = {"40x": "20x", "20x": "10x", "10x": "5x"}


def adjacent_scale_config() -> dict:
    return retrieval_config().get("adjacent_scale", {})


def parent_zoom_for(level: str) -> str | None:
    """Parent zoom tier for CMT adjacent-scale enrichment, or None at coarsest level."""
    cfg = adjacent_scale_config()
    if not cfg.get("enabled", True):
        return None
    parent_map = cfg.get("parent_map", _DEFAULT_PARENT_MAP)
    parent = parent_map.get(normalize_zoom(level))
    return normalize_zoom(parent) if parent else None


def include_grandparent(*, tier: str | None = None, node_kind: str | None = None) -> bool:
    """Whether to attach a second ancestor (parent's parent) for integration nodes."""
    cfg = adjacent_scale_config()
    if not cfg.get("grandparent_for_integration", True):
        return False
    tier_key = (tier or "").lower()
    kind_key = (node_kind or "").lower()
    return tier_key == "integration" or kind_key in ("integration", "report")
