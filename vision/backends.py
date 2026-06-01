"""Visual evidence providers for the VLM (P1: thumbnail, no TITAN)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from graph.schema import Node
from vision.cache import SlideCache


@dataclass
class VisualBundle:
    """What the AnswerBackend receives as visual input for one graph step."""

    thumbnail_path: Path | None = None
    patch_paths: list[Path] = field(default_factory=list)
    patch_embeddings: Any = None
    slide_embedding: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VisualEvidenceProvider(Protocol):
    def for_node(
        self,
        node: Node,
        slide_cache: SlideCache | None,
        *,
        query: str,
        retriever: Any = None,
    ) -> VisualBundle: ...


def get_visual_provider(method: str, cache_root: Path | None = None) -> VisualEvidenceProvider:
    if method == "thumbnail":
        from vision.thumbnail import ThumbnailProvider

        return ThumbnailProvider(cache_root=cache_root)
    if method == "none":
        from vision.thumbnail import NoneVisualProvider

        return NoneVisualProvider()
    if method == "patch_retrieve":
        from vision.thumbnail import PatchRetrieveProvider

        return PatchRetrieveProvider(cache_root=cache_root)
    if method == "slide_embed":
        from vision.slide_embed import SlideEmbedProvider

        return SlideEmbedProvider(cache_root=cache_root)
    raise ValueError(f"Unknown visual provider: {method!r}")
