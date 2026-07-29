"""Graph-guided retrieval: maps node.zoom_level → magnification-specific patch pool."""

from __future__ import annotations

from pathlib import Path

from retrieval.base import PatchRetriever
from retrieval.titan_cosine import RetrievedPatch
from vision.cache import SlideCache
from vision.mag_config import fixed_retrieval_pool, top_k_for_zoom


class GraphGuidedRetriever:
    """Delegates to inner retriever; retrieval pool is fixed (default: 20x)."""

    def __init__(self, inner: PatchRetriever):
        self.inner = inner

    def retrieve(
        self,
        query: str,
        slide_cache: SlideCache,
        *,
        level: str | None = None,
        k: int | None = None,
        exclude: set[int] | None = None,
        wsi_path: Path | None = None,
        return_images: bool = True,
        tier: str | None = None,
        node_kind: str | None = None,
        anchor_coord_lv0: tuple[int, int] | None = None,
        min_dist_pool_px: int = 0,
        **kwargs,
    ) -> list[RetrievedPatch]:
        level = fixed_retrieval_pool() if level is None else level
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
            tier=tier,
            node_kind=node_kind,
            anchor_coord_lv0=anchor_coord_lv0,
            min_dist_pool_px=min_dist_pool_px,
            **kwargs,
        )
