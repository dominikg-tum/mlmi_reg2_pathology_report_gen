"""TITAN text-guided cosine retrieval.

Given a text query (question + memory context) return the top-K patch embeddings
for one slide. TITAN's text and image encoders share a 768-dim space, so cosine
similarity between a query vector and patch embeddings is semantically meaningful.

Built FIRST and reused by both training-sample construction (G1) and the agent
loop (G2) -> guarantees train/test consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SlideEmbeddings:
    """Cached output of WP2 patch encoding for one slide."""

    embeddings: np.ndarray  # [N_patches, 768]
    coords: np.ndarray      # [N_patches, 2] spatial (x, y)


class TitanRetriever:
    def __init__(self, text_encoder=None):
        # text_encoder: frozen TITAN text encoder; injected so tests can stub it.
        self.text_encoder = text_encoder

    def encode_query(self, query: str) -> np.ndarray:
        if self.text_encoder is None:
            raise RuntimeError("TITAN text encoder not loaded")
        vec = self.text_encoder(query)  # [768]
        return np.asarray(vec, dtype=np.float32)

    def retrieve(
        self,
        query: str,
        slide: SlideEmbeddings,
        *,
        level: str = "high",
        k: int = 3,
        exclude: set[int] | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        """Return (top-K patch embeddings, their indices).

        `level` is a hook for dual-scale retrieval (low-mag for global nodes,
        high-mag for local nodes). `exclude` supports the G2 low-confidence retry.
        """
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
