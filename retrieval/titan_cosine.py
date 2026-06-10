"""TITAN text-guided cosine retrieval over K-means centroid pool (Phases 1, 2, 4)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vision.cache import SlideCache
from vision.mag_config import normalize_zoom, retrieval_config, top_k_for_zoom
from vision.wsi_io import find_parent_patch_index, load_patch_at_coord


@dataclass
class RetrievedPatch:
    """One retrieved patch plus optional ×10 parent (Phase 2)."""

    patch_image: object | None
    parent_image: object | None
    level: str
    coord: tuple[int, int]
    parent_coord: tuple[int, int] | None
    similarity: float
    index: int
    parent_index: int | None = None


@dataclass
class SlideEmbeddings:
    embeddings: np.ndarray
    coords: np.ndarray
    patch_size_lv0: int


class TitanCosineRetriever:
    def __init__(self, text_encoder=None, *, top_k: int | None = None, d_min_20x: int | None = None):
        self.text_encoder = text_encoder
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

    def retrieve(
        self,
        query: str,
        slide_cache: SlideCache,
        *,
        level: str = "20x",
        k: int | None = None,
        exclude: set[int] | None = None,
        wsi_path: Path | None = None,
        return_images: bool = True,
    ) -> list[RetrievedPatch]:
        level = normalize_zoom(level)
        k = k if k is not None else top_k_for_zoom(level)
        slide = self._load_embeddings(slide_cache, level)
        centroid_idx = self._load_centroid_indices(slide_cache, level)

        centroid_emb = slide.embeddings[centroid_idx]
        centroid_coords = slide.coords[centroid_idx]

        q = self.encode_query(query)
        sims = _cosine(q, centroid_emb)
        if exclude:
            sims[list(exclude)] = -np.inf

        order = np.argsort(-sims)
        ps_lv0 = slide.patch_size_lv0 or self._load_meta_patch_size(slide_cache, level)
        if ps_lv0 <= 0:
            raise RuntimeError(f"patch_size_lv0 missing in meta_{level}.json")

        accepted_local = self._diversity_filter_with_size(
            order, centroid_coords, level=level, k=k, patch_size_lv0=ps_lv0
        )

        parent_coords_medium: np.ndarray | None = None
        ps_medium = 0
        if level == "20x":
            medium = self._load_embeddings(slide_cache, "10x")
            parent_coords_medium = medium.coords
            ps_medium = medium.patch_size_lv0 or self._load_meta_patch_size(
                slide_cache, "10x"
            )

        results: list[RetrievedPatch] = []
        for local_i in accepted_local:
            global_i = int(centroid_idx[local_i])
            coord = tuple(int(centroid_coords[local_i, 0]), int(centroid_coords[local_i, 1]))
            sim = float(sims[local_i])
            parent_coord = None
            parent_index = None
            parent_image = None

            if level == "20x" and parent_coords_medium is not None and ps_medium > 0:
                parent_index = find_parent_patch_index(
                    coord,
                    parent_coords_medium,
                    patch_size_lv0_high=ps_lv0,
                    patch_size_lv0_medium=ps_medium,
                )
                parent_coord = tuple(
                    int(parent_coords_medium[parent_index, 0]),
                    int(parent_coords_medium[parent_index, 1]),
                )

            patch_image = None
            if return_images and wsi_path is not None:
                from vision.mag_config import mag_band_config

                objective, patch_size = mag_band_config(level)
                patch_image = load_patch_at_coord(
                    wsi_path, coord, objective=objective, patch_size=patch_size
                )
                if parent_coord is not None:
                    obj_m, ps_m = mag_band_config("10x")
                    parent_image = load_patch_at_coord(
                        wsi_path, parent_coord, objective=obj_m, patch_size=ps_m
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
