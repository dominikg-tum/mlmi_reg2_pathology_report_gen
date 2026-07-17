from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from graph.schema import InteractionType, Node, NodeKind, Tier, VisualPolicy, ZoomLevel
from vision.thumbnail import PatchRetrieveProvider


@dataclass
class _DummyRetriever:
    last_level: str | None = None

    def retrieve(self, query, slide_cache, *, level="20x", **kwargs):
        self.last_level = level
        return []


def test_patch_retrieve_provider_uses_fixed_pool(monkeypatch, tmp_path: Path):
    from vision.cache import SlideCache

    # zoom_level is just a hint, must not route retrieval pools anymore
    node = Node(
        id="n1",
        label="n1",
        question="q",
        tier=Tier.LOCAL_FEATURES,
        node_kind=NodeKind.LOCAL,
        interaction=InteractionType.SINGLE_SELECT,
        options=["a"],
        edges={"a": "n2"},
        zoom_level=ZoomLevel.X10,
        visual_policy=VisualPolicy.PATCH_RETRIEVE,
    )

    slide_cache = SlideCache(slide_id="case01.svs", cache_dir=tmp_path, thumbnail_path=None)
    retriever = _DummyRetriever()

    provider = PatchRetrieveProvider(cache_root=tmp_path)
    _ = provider.for_node(node, slide_cache, query="q", retriever=retriever)
    assert retriever.last_level == "20x"

