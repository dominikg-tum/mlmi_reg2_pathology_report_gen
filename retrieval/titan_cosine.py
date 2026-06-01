"""TITAN text-guided cosine retrieval over offline patch embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vision.cache import SlideCache


@dataclass
class SlideEmbeddings:
    """In-memory patch matrix loaded from offline cache."""

    embeddings: np.ndarray
    coords: np.ndarray


class TitanCosineRetriever:
    def __init__(self, text_encoder=None):
        self.text_encoder = text_encoder

    def encode_query(self, query: str) -> np.ndarray:
        if self.text_encoder is None:
            raise RuntimeError("TITAN text encoder not loaded — wire in P2")
        vec = self.text_encoder(query)
        return np.asarray(vec, dtype=np.float32)

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
            emb = data.get("embeddings", data.get("emb"))
            coords = data.get("coords", np.zeros((len(emb), 2)))
        else:
            emb = data
            coords = np.zeros((len(emb), 2))
        return SlideEmbeddings(
            embeddings=np.asarray(emb, dtype=np.float32),
            coords=np.asarray(coords),
        )

    def retrieve(
        self,
        query: str,
        slide_cache: SlideCache,
        *,
        level: str = "high",
        k: int = 3,
        exclude: set[int] | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        slide = self._load_embeddings(slide_cache, level)
        q = self.encode_query(query)
        emb = slide.embeddings
        sims = _cosine(q, emb)
        if exclude:
            sims[list(exclude)] = -np.inf
        top = np.argsort(-sims)[:k]
        return emb[top], top.tolist()


def _cosine(q: np.ndarray, m: np.ndarray) -> np.ndarray:
    q = q / (np.linalg.norm(q) + 1e-8)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)
    return m @ q
