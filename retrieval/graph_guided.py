"""Graph-guided retrieval: maps node.retrieval_level → offline cache band."""

from __future__ import annotations

from pathlib import Path

from retrieval.base import PatchRetriever
from retrieval.titan_cosine import RetrievedPatch
from vision.cache import SlideCache


class GraphGuidedRetriever:
    """Delegates to inner retriever; level comes from the graph node."""

    def __init__(self, inner: PatchRetriever):
        self.inner = inner

    def retrieve(
        self,
        query: str,
        slide_cache: SlideCache,
        *,
        level: str = "high",
        k: int = 3,
        exclude: set[int] | None = None,
        wsi_path: Path | None = None,
        return_images: bool = True,
    ) -> list[RetrievedPatch]:
        return self.inner.retrieve(
            query,
            slide_cache,
            level=level,
            k=k,
            exclude=exclude,
            wsi_path=wsi_path,
            return_images=return_images,
        )
