"""Slide-level HSV tissue mask for offline tiling (Phase B default)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SlideTissueMask:
    """Binary tissue mask scaled to level-0 WSI coordinates."""

    mask: np.ndarray  # uint8, 0=background 255=tissue
    level0_width: int
    level0_height: int

    def tissue_fraction(self, x0: int, y0: int, patch_size_lv0: int) -> float:
        """Fraction of mask pixels marked tissue inside a level-0 patch bbox."""
        if self.mask.size == 0 or patch_size_lv0 <= 0:
            return 0.0
        mw, mh = self.mask.shape[1], self.mask.shape[0]
        if self.level0_width <= 0 or self.level0_height <= 0:
            return 0.0

        sx = mw / float(self.level0_width)
        sy = mh / float(self.level0_height)
        x1 = int(x0 * sx)
        y1 = int(y0 * sy)
        x2 = min(int((x0 + patch_size_lv0) * sx), mw)
        y2 = min(int((y0 + patch_size_lv0) * sy), mh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        region = self.mask[y1:y2, x1:x2]
        return float((region > 0).mean())


def rgb_to_hsv01(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB uint8 (H, W, 3) -> H, S, V in [0, 1]."""
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    cmax = np.max(rgb_f, axis=-1)
    cmin = np.min(rgb_f, axis=-1)
    delta = cmax - cmin

    hue = np.zeros_like(cmax)
    mask = delta > 1e-6
    rc = (cmax - r) / (delta + 1e-8)
    gc = (cmax - g) / (delta + 1e-8)
    bc = (cmax - b) / (delta + 1e-8)
    hue = np.where((cmax == r) & mask, (bc - gc) % 6.0, hue)
    hue = np.where((cmax == g) & mask, (rc - bc) + 2.0, hue)
    hue = np.where((cmax == b) & mask, (gc - rc) + 4.0, hue)
    hue = (hue / 6.0) % 1.0

    sat = np.where(cmax > 1e-6, delta / (cmax + 1e-8), 0.0)
    val = cmax
    return hue, sat, val


def build_hsv_tissue_mask(
    rgb: np.ndarray,
    *,
    sat_min: float = 0.08,
    val_max: float = 0.95,
    morph_close_px: int = 5,
) -> np.ndarray:
    """Return uint8 mask (255=tissue) from an RGB thumbnail/low-res image."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB HxWx3, got shape {rgb.shape}")
    _, sat, val = rgb_to_hsv01(rgb)
    tissue = (sat >= float(sat_min)) & (val <= float(val_max))
    mask = (tissue.astype(np.uint8)) * 255
    if morph_close_px > 0:
        mask = _morph_close_binary(mask, kernel_px=int(morph_close_px))
    return mask


def _morph_close_binary(mask: np.ndarray, *, kernel_px: int) -> np.ndarray:
    """3x3 dilation then erosion — cheap stand-in for scipy binary close."""
    k = max(int(kernel_px), 1)
    if k <= 1:
        return mask
    dilated = _max_filter(mask > 0, k)
    closed = _min_filter(dilated, k)
    return (closed.astype(np.uint8)) * 255


def _max_filter(binary: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    padded = np.pad(binary.astype(np.uint8), pad, mode="constant", constant_values=0)
    h, w = binary.shape
    out = np.zeros_like(binary, dtype=bool)
    for dy in range(k):
        for dx in range(k):
            out |= padded[dy : dy + h, dx : dx + w].astype(bool)
    return out


def _min_filter(binary: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    padded = np.pad(binary.astype(np.uint8), pad, mode="constant", constant_values=1)
    h, w = binary.shape
    out = np.ones_like(binary, dtype=bool)
    for dy in range(k):
        for dx in range(k):
            out &= padded[dy : dy + h, dx : dx + w].astype(bool)
    return out


def build_mask_from_thumbnail(
    svs_path: Path,
    *,
    max_edge_px: int = 1024,
    sat_min: float = 0.08,
    val_max: float = 0.95,
    morph_close_px: int = 5,
) -> SlideTissueMask:
    """Build a slide-level mask from an openslide pyramid thumbnail."""
    import openslide
    from PIL import Image

    slide = openslide.OpenSlide(str(svs_path))
    try:
        thumb = slide.get_thumbnail((max_edge_px, max_edge_px)).convert("RGB")
        level0_w, level0_h = slide.dimensions
        rgb = np.asarray(thumb)
        mask = build_hsv_tissue_mask(
            rgb,
            sat_min=sat_min,
            val_max=val_max,
            morph_close_px=morph_close_px,
        )
        return SlideTissueMask(mask=mask, level0_width=level0_w, level0_height=level0_h)
    finally:
        slide.close()


def save_tissue_mask_png(mask: SlideTissueMask, out_path: Path) -> None:
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.mask).save(out_path)


def load_tissue_mask_png(
    mask_path: Path,
    *,
    level0_width: int,
    level0_height: int,
) -> SlideTissueMask:
    from PIL import Image

    arr = np.asarray(Image.open(mask_path).convert("L"))
    return SlideTissueMask(
        mask=(arr > 0).astype(np.uint8) * 255,
        level0_width=level0_width,
        level0_height=level0_height,
    )


def make_patch_accept_fn(
    *,
    method: str,
    background_threshold: int = 220,
    min_tissue_fraction: float = 0.40,
    tissue_mask: SlideTissueMask | None = None,
) -> Callable[[np.ndarray, int, int, int], bool]:
    """Return a predicate(arr, x0, y0, patch_size_lv0) for iter_tissue_patches."""

    def _mean_threshold(arr: np.ndarray, _x0: int, _y0: int, _ps: int) -> bool:
        gray = arr.mean(axis=2) if arr.ndim == 3 else arr
        return float(gray.mean()) <= float(background_threshold)

    if method == "mean_threshold":
        return _mean_threshold

    if method == "slide_mask":
        if tissue_mask is None:
            raise ValueError("slide_mask method requires a SlideTissueMask")

        def _slide_mask(_arr: np.ndarray, x0: int, y0: int, ps: int) -> bool:
            frac = tissue_mask.tissue_fraction(x0, y0, ps)
            return frac >= float(min_tissue_fraction)

        return _slide_mask

    raise ValueError(f"Unknown tissue_filter.method {method!r}")
