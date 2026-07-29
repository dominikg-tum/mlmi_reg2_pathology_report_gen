"""TITAN text-guided cosine retrieval over the 20x CONCH pool (Phases 1, 2, 4)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vision.cache import SlideCache
from vision.mag_config import (
    default_search_all_patches,
    include_grandparent,
    mag_band_config,
    normalize_zoom,
    parent_zoom_for,
    retrieval_config,
    top_k_for_zoom,
)
from vision.wsi_io import find_parent_patch_index, load_patch_at_coord


@dataclass
class RetrievedPatch:
    """One retrieved patch plus optional adjacent-scale ancestors (CMT enricher)."""

    patch_image: object | None
    parent_image: object | None
    level: str
    coord: tuple[int, int]
    parent_coord: tuple[int, int] | None
    similarity: float
    index: int
    parent_index: int | None = None
    parent_level: str | None = None
    grandparent_image: object | None = None
    grandparent_coord: tuple[int, int] | None = None
    grandparent_index: int | None = None
    grandparent_level: str | None = None


@dataclass
class SlideEmbeddings:
    embeddings: np.ndarray
    coords: np.ndarray
    patch_size_lv0: int


class TitanCosineRetriever:
    def __init__(
        self,
        text_encoder=None,
        *,
        top_k: int | None = None,
        d_min_20x: int | None = None,
        search_all_patches: bool | None = None,
    ):
        self.text_encoder = text_encoder
        self.search_all_patches = (
            search_all_patches
            if search_all_patches is not None
            else default_search_all_patches()
        )
        rcfg = retrieval_config()
        self.top_k = top_k if top_k is not None else int(rcfg.get("top_k", 5))
        self.d_min_20x = d_min_20x if d_min_20x is not None else int(rcfg.get("d_min_20x_px", 512))

    def encode_query(self, query: str) -> np.ndarray:
        if self.text_encoder is None:
            raise RuntimeError("TITAN text encoder not loaded — wire TitanEncoder.encode_text")
        vec = self.text_encoder(query)
        return np.asarray(vec, dtype=np.float32)

    def _load_meta_patch_size(self, slide_cache: SlideCache, level: str) -> int:
        meta_path = slide_cache.meta_path_for_level(level)
        if meta_path and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            return int(meta.get("patch_size_lv0", 0))
        return 0

    def _load_embeddings(self, slide_cache: SlideCache, level: str) -> SlideEmbeddings:
        path = slide_cache.embedding_path_for_level(level)
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"No offline embeddings for level={level!r} at {path}. "
                "Run scripts/vision/encode_patches_offline.py first."
            )
        import torch

        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            emb = np.asarray(data.get("embeddings", data.get("emb")), dtype=np.float32)
            coords = np.asarray(data.get("coords", np.zeros((len(emb), 2))))
        else:
            emb = np.asarray(data, dtype=np.float32)
            coords = np.zeros((len(emb), 2))

        coord_path = slide_cache.coords_path_for_level(level)
        if coord_path and coord_path.exists():
            coords = np.asarray(torch.load(coord_path, map_location="cpu", weights_only=False))

        ps_lv0 = self._load_meta_patch_size(slide_cache, level)
        return SlideEmbeddings(embeddings=emb, coords=coords, patch_size_lv0=ps_lv0)

    def _load_centroid_indices(self, slide_cache: SlideCache, level: str) -> np.ndarray:
        path = slide_cache.centroid_path_for_level(level)
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"No K-means centroids for level={level!r} at {path}. "
                "Run scripts/vision/build_kmeans_index.py first."
            )
        import torch

        indices = torch.load(path, map_location="cpu", weights_only=False)
        return np.asarray(indices, dtype=np.int64)

    def _resolve_ancestor(
        self,
        coord: tuple[int, int],
        slide_cache: SlideCache,
        *,
        child_level: str,
        parent_level: str,
        child_patch_size_lv0: int,
    ) -> tuple[int, tuple[int, int]] | None:
        parent = self._load_embeddings(slide_cache, parent_level)
        ps_parent = parent.patch_size_lv0 or self._load_meta_patch_size(slide_cache, parent_level)
        if ps_parent <= 0:
            return None
        parent_index = find_parent_patch_index(
            coord,
            parent.coords,
            patch_size_lv0_high=child_patch_size_lv0,
            patch_size_lv0_medium=ps_parent,
        )
        parent_coord = (
            int(parent.coords[parent_index, 0]),
            int(parent.coords[parent_index, 1]),
        )
        return parent_index, parent_coord

    def _load_patch_image(
        self, wsi_path: Path, coord: tuple[int, int], level: str
    ) -> object:
        objective, patch_size = mag_band_config(level)
        return load_patch_at_coord(
            wsi_path, coord, objective=objective, patch_size=patch_size
        )

    def retrieve(
        self,
        query: str,
        slide_cache: SlideCache,
        *,
        level: str = "20x",
        k: int | None = None,
        exclude: set[int] | None = None,
        anchor_coord_lv0: tuple[int, int] | None = None,
        min_dist_pool_px: int = 0,
        wsi_path: Path | None = None,
        return_images: bool = True,
        tier: str | None = None,
        node_kind: str | None = None,
    ) -> list[RetrievedPatch]:
        level = normalize_zoom(level)
        k = k if k is not None else top_k_for_zoom(level)
        slide = self._load_embeddings(slide_cache, level)

        if self.search_all_patches:
            pool_emb = slide.embeddings
            pool_coords = slide.coords
            pool_indices = np.arange(len(slide.embeddings), dtype=np.int64)
        else:
            centroid_idx = self._load_centroid_indices(slide_cache, level)
            pool_emb = slide.embeddings[centroid_idx]
            pool_coords = slide.coords[centroid_idx]
            pool_indices = centroid_idx

        q = self.encode_query(query)
        sims = _cosine(q, pool_emb)
        if exclude:
            sims[np.isin(pool_indices, list(exclude))] = -np.inf

        ps_lv0 = slide.patch_size_lv0 or self._load_meta_patch_size(slide_cache, level)
        if ps_lv0 <= 0:
            raise RuntimeError(f"patch_size_lv0 missing in meta_{level}.json")

        # Optional strict min-distance filter (paired_regions fallback).
        # min_dist_pool_px is configured in pool-magnification pixels
        # (retrieval.paired_regions.min_dist_20x_px), while coords are level-0.
        # On a 40x-scanned slide one 20x pixel is two level-0 pixels.
        if anchor_coord_lv0 is not None and min_dist_pool_px > 0:
            _, native_px = mag_band_config(level)
            lv0_per_pool_px = (ps_lv0 / float(native_px)) if native_px > 0 else 1.0
            min_dist_lv0 = float(min_dist_pool_px) * lv0_per_pool_px
            ax = float(anchor_coord_lv0[0] + ps_lv0 / 2.0)
            ay = float(anchor_coord_lv0[1] + ps_lv0 / 2.0)
            centres = pool_coords.astype(np.float64) + (ps_lv0 / 2.0)
            dists = np.sqrt((centres[:, 0] - ax) ** 2 + (centres[:, 1] - ay) ** 2)
            sims[dists < min_dist_lv0] = -np.inf
            if not np.isfinite(sims).any():
                # Relax once.
                relax = min_dist_lv0 / 2.0
                sims = _cosine(q, pool_emb)
                if exclude:
                    sims[np.isin(pool_indices, list(exclude))] = -np.inf
                if relax > 0:
                    sims[dists < relax] = -np.inf

        order = np.argsort(-sims)
        order = order[sims[order] != -np.inf]
        accepted_local = self._diversity_filter_with_size(
            order, pool_coords, level=level, k=k, patch_size_lv0=ps_lv0
        )

        parent_level = parent_zoom_for(level)
        want_grandparent = (
            parent_level is not None
            and include_grandparent(tier=tier, node_kind=node_kind)
        )
        grandparent_level = parent_zoom_for(parent_level) if want_grandparent else None

        results: list[RetrievedPatch] = []
        for local_i in accepted_local:
            global_i = int(pool_indices[local_i])
            coord = (int(pool_coords[local_i, 0]), int(pool_coords[local_i, 1]))
            sim = float(sims[local_i])
            parent_coord = None
            parent_index = None
            parent_image = None
            grandparent_coord = None
            grandparent_index = None
            grandparent_image = None

            if parent_level is not None:
                resolved = self._resolve_ancestor(
                    coord,
                    slide_cache,
                    child_level=level,
                    parent_level=parent_level,
                    child_patch_size_lv0=ps_lv0,
                )
                if resolved is not None:
                    parent_index, parent_coord = resolved
                    if grandparent_level is not None:
                        ps_parent = self._load_meta_patch_size(slide_cache, parent_level)
                        if ps_parent <= 0:
                            ps_parent = self._load_embeddings(
                                slide_cache, parent_level
                            ).patch_size_lv0
                        gp = self._resolve_ancestor(
                            parent_coord,
                            slide_cache,
                            child_level=parent_level,
                            parent_level=grandparent_level,
                            child_patch_size_lv0=ps_parent,
                        )
                        if gp is not None:
                            grandparent_index, grandparent_coord = gp

            patch_image = None
            if return_images and wsi_path is not None:
                patch_image = self._load_patch_image(wsi_path, coord, level)
                if parent_coord is not None and parent_level is not None:
                    parent_image = self._load_patch_image(
                        wsi_path, parent_coord, parent_level
                    )
                if grandparent_coord is not None and grandparent_level is not None:
                    grandparent_image = self._load_patch_image(
                        wsi_path, grandparent_coord, grandparent_level
                    )

            results.append(
                RetrievedPatch(
                    patch_image=patch_image,
                    parent_image=parent_image,
                    level=level,
                    coord=coord,
                    parent_coord=parent_coord,
                    similarity=sim,
                    index=global_i,
                    parent_index=parent_index,
                    parent_level=parent_level,
                    grandparent_image=grandparent_image,
                    grandparent_coord=grandparent_coord,
                    grandparent_index=grandparent_index,
                    grandparent_level=grandparent_level,
                )
            )
        return results

    def _diversity_filter_with_size(
        self,
        order: np.ndarray,
        coords: np.ndarray,
        *,
        level: str,
        k: int,
        patch_size_lv0: int,
    ) -> list[int]:
        d_min = float(self.d_min_20x if level == "20x" else self.d_min_20x * 2)
        accepted: list[int] = []
        half = patch_size_lv0 / 2.0
        centres = coords.astype(np.float64) + half

        for idx in order:
            c = centres[int(idx)]
            if all(np.linalg.norm(c - centres[a]) > d_min for a in accepted):
                accepted.append(int(idx))
            if len(accepted) >= k:
                break
        return accepted


def _cosine(q: np.ndarray, m: np.ndarray) -> np.ndarray:
    q = q / (np.linalg.norm(q) + 1e-8)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)
    return m @ q
