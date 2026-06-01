"""Graph-guided retrieval: maps node.retrieval_level → offline cache band."""

from __future__ import annotations

from retrieval.base import PatchRetriever
from vision.cache import SlideCache


class GraphGuidedRetriever:
    """Delegates to inner retriever; level comes from the graph node (MST substitute)."""

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
    ) -> tuple:
        return self.inner.retrieve(
            query, slide_cache, level=level, k=k, exclude=exclude
        )
