"""P1 default: whole-slide thumbnail (no TITAN)."""

from __future__ import annotations

from pathlib import Path

from graph.schema import Node, VisualPolicy
from vision.backends import VisualBundle
from vision.cache import SlideCache


class ThumbnailProvider:
    def __init__(self, cache_root: Path | None = None):
        self.cache_root = cache_root

    def for_node(
        self,
        node: Node,
        slide_cache: SlideCache | None,
        *,
        query: str,
        retriever=None,
    ) -> VisualBundle:
        thumb = slide_cache.thumbnail_path if slide_cache else None
        bundle = VisualBundle(thumbnail_path=thumb, metadata={"visual": "thumbnail"})
        if node.visual_policy == VisualPolicy.BOTH and retriever is not None and slide_cache:
            try:
                from retrieval.graph_guided import GraphGuidedRetriever

                if isinstance(retriever, GraphGuidedRetriever):
                    emb, idx = retriever.retrieve(
                        query,
                        slide_cache,
                        level=node.retrieval_level_str,
                    )
                    bundle.patch_embeddings = emb
                    bundle.metadata["patch_indices"] = idx
            except (RuntimeError, NotImplementedError):
                pass
        return bundle


class NoneVisualProvider:
    """Text-only debug / oracle runs."""

    def for_node(self, node, slide_cache=None, *, query: str, retriever=None) -> VisualBundle:
        return VisualBundle(metadata={"visual": "none"})


class PatchRetrieveProvider:
    """P2: top-K patches from offline cache via retriever."""

    def __init__(self, cache_root: Path | None = None):
        self.cache_root = cache_root

    def for_node(
        self,
        node: Node,
        slide_cache: SlideCache | None,
        *,
        query: str,
        retriever=None,
    ) -> VisualBundle:
        bundle = VisualBundle(metadata={"visual": "patch_retrieve"})
        if slide_cache and slide_cache.thumbnail_path:
            bundle.thumbnail_path = slide_cache.thumbnail_path
        if retriever is None or slide_cache is None:
            return bundle
        if not node.needs_patch_retrieval():
            return bundle
        emb, idx = retriever.retrieve(
            query,
            slide_cache,
            level=node.retrieval_level_str,
        )
        bundle.patch_embeddings = emb
        bundle.metadata["patch_indices"] = idx
        return bundle
