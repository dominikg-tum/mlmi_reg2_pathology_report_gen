"""Magnification navigation — graph-guided default; MMNavAgent hook for later."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from graph.schema import Node
from vision.backends import VisualBundle, get_visual_provider
from vision.cache import SlideCache
from vision.graph_visual import GraphPolicyVisualProvider


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
    """Uses node.zoom_level + node.visual_policy (graph-as-MST)."""

    def __init__(
        self,
        visual_method: str = "thumbnail",
        cache_root=None,
        *,
        wsi_path: Path | None = None,
        wsi_data_dir: Path | None = None,
    ):
        if visual_method == "patch_retrieve":
            self._visual = GraphPolicyVisualProvider(
                cache_root,
                wsi_path=wsi_path,
                wsi_data_dir=wsi_data_dir,
            )
        else:
            self._visual = get_visual_provider(
                visual_method,
                cache_root=cache_root,
                wsi_path=wsi_path,
                wsi_data_dir=wsi_data_dir,
            )

    def select_visual_bundle(
        self,
        node: Node,
        slide_cache: SlideCache | None,
        memory: list,
        *,
        query: str,
        retriever=None,
    ) -> VisualBundle:
        _ = memory
        return self._visual.for_node(
            node, slide_cache, query=query, retriever=retriever
        )


def get_navigator(method: str, **kwargs) -> MagnificationNavigator:
    if method == "graph_guided":
        return GraphGuidedNavigator(
            visual_method=kwargs.get("visual", "thumbnail"),
            cache_root=kwargs.get("cache_root"),
            wsi_path=kwargs.get("wsi_path"),
            wsi_data_dir=kwargs.get("wsi_data_dir"),
        )
    if method == "mnavagent":
        raise NotImplementedError(
            "Plug in Han's MMNavAgent when code is public; see arXiv:2603.02079"
        )
    raise ValueError(f"Unknown navigator: {method!r}")
