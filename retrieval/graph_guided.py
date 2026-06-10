"""Graph-guided retrieval: maps node.zoom_level → magnification-specific patch pool."""

from __future__ import annotations

from pathlib import Path

from retrieval.base import PatchRetriever
from retrieval.titan_cosine import RetrievedPatch
from vision.cache import SlideCache
from vision.mag_config import top_k_for_zoom


class GraphGuidedRetriever:
    """Delegates to inner retriever; mag band comes from node.zoom_level (not a flat pool)."""

    def __init__(self, inner: PatchRetriever):
        self.inner = inner

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
        if k is None:
            k = top_k_for_zoom(level)
        return self.inner.retrieve(
            query,
            slide_cache,
            level=level,
            k=k,
            exclude=exclude,
            wsi_path=wsi_path,
            return_images=return_images,
        )
