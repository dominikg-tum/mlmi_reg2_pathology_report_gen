"""Optional ablation: single precomputed slide-level embedding."""

from __future__ import annotations

from pathlib import Path

from graph.schema import Node
from vision.backends import VisualBundle
from vision.cache import SlideCache


class SlideEmbedProvider:
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
        # DOMI: load slide_embedding.pt from slide_cache dir when implemented
        return VisualBundle(
            thumbnail_path=slide_cache.thumbnail_path if slide_cache else None,
            metadata={"visual": "slide_embed", "todo": "load slide_embedding.pt"},
        )
