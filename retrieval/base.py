"""Pluggable patch retriever protocol (not TITAN-only)."""

from __future__ import annotations

from typing import Any, Protocol

from vision.cache import SlideCache


class PatchRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        slide_cache: SlideCache,
        *,
        level: str = "high",
        k: int = 3,
        exclude: set[int] | None = None,
    ) -> tuple[Any, list[int]]: ...


def get_retriever(method: str, **kwargs) -> PatchRetriever | None:
    if method in ("none", ""):
        return None
    if method == "titan_cosine":
        from retrieval.titan_cosine import TitanCosineRetriever

        return TitanCosineRetriever(text_encoder=kwargs.get("text_encoder"))
    if method == "graph_guided":
        from retrieval.graph_guided import GraphGuidedRetriever

        inner = get_retriever(kwargs.get("inner", "titan_cosine"), **kwargs)
        if inner is None:
            raise ValueError("graph_guided requires inner retriever (e.g. titan_cosine)")
        return GraphGuidedRetriever(inner)
    raise ValueError(f"Unknown retriever: {method!r}")
