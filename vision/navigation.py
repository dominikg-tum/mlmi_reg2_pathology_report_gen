"""Magnification navigation — graph-guided default; MMNavAgent hook for later."""

from __future__ import annotations

from typing import Protocol

from graph.schema import Node
from vision.backends import VisualBundle, get_visual_provider
from vision.cache import SlideCache


class MagnificationNavigator(Protocol):
    def select_visual_bundle(
        self,
        node: Node,
        slide_cache: SlideCache | None,
        memory: list,
        *,
        query: str,
        retriever=None,
    ) -> VisualBundle: ...


class GraphGuidedNavigator:
    """Uses node.retrieval_level + visual_policy (graph-as-MST)."""

    def __init__(self, visual_method: str = "thumbnail", cache_root=None):
        self._visual = get_visual_provider(visual_method, cache_root=cache_root)

    def select_visual_bundle(
        self,
        node: Node,
        slide_cache: SlideCache | None,
        memory: list,
        *,
        query: str,
        retriever=None,
    ) -> VisualBundle:
        return self._visual.for_node(
            node, slide_cache, query=query, retriever=retriever
        )


def get_navigator(method: str, **kwargs) -> MagnificationNavigator:
    if method == "graph_guided":
        return GraphGuidedNavigator(
            visual_method=kwargs.get("visual", "thumbnail"),
            cache_root=kwargs.get("cache_root"),
        )
    if method == "mnavagent":
        raise NotImplementedError(
            "Plug in Han's MMNavAgent when code is public; see arXiv:2603.02079"
        )
    raise ValueError(f"Unknown navigator: {method!r}")
