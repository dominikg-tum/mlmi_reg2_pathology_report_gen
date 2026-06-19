from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

from graph import GRAPH
from graph.schema import VisualPolicy
from vision.cache import SlideCache
from vision.graph_visual import GraphPolicyVisualProvider
from vision.navigation import GraphGuidedNavigator


@dataclass
class _FakeRetrieved:
    patch_image: object | None = None
    parent_image: object | None = None
    grandparent_image: object | None = None
    level: str = "20x"
    coord: tuple[int, int] = (0, 0)
    parent_coord: tuple[int, int] | None = None
    similarity: float = 0.9
    index: int = 0
    parent_index: int | None = None
    parent_level: str | None = None
    grandparent_coord: tuple[int, int] | None = None
    grandparent_index: int | None = None
    grandparent_level: str | None = None


class _FakePatchImage:
    def save(self, path):
        Path(path).write_bytes(b"png")


def _slide_cache(tmp_path: Path) -> SlideCache:
    slide_dir = tmp_path / "TUM_Uterus_0001.svs"
    slide_dir.mkdir(exist_ok=True)
    thumb = slide_dir / "thumbnail.png"
    thumb.write_bytes(b"png")
    return SlideCache(
        slide_id="TUM_Uterus_0001.svs",
        cache_dir=slide_dir,
        thumbnail_path=thumb,
    )


def test_thumbnail_only_node_gets_thumbnail(tmp_path: Path):
    provider = GraphPolicyVisualProvider()
    node = GRAPH["organ_procedure"]
    assert node.visual_policy == VisualPolicy.THUMBNAIL_ONLY

    bundle = provider.for_node(node, _slide_cache(tmp_path), query="q")

    assert bundle.thumbnail_path is not None
    assert bundle.patch_paths == []
    assert bundle.metadata["visual_policy"] == "thumbnail_only"


def test_patch_retrieve_node_gets_patches_not_thumbnail(tmp_path: Path):
    provider = GraphPolicyVisualProvider()
    node = GRAPH["endometrium_assessment"]
    slide_cache = _slide_cache(tmp_path)
    retriever = MagicMock()
    retriever.retrieve.return_value = [_FakeRetrieved(patch_image=_FakePatchImage())]

    bundle = provider.for_node(
        node, slide_cache, query="q", retriever=retriever
    )

    assert bundle.thumbnail_path is None
    assert len(bundle.patch_paths) == 1
    assert bundle.metadata["visual_policy"] == "patch_retrieve"
    retriever.retrieve.assert_called_once()


def test_both_node_gets_thumbnail_and_patches(tmp_path: Path):
    provider = GraphPolicyVisualProvider()
    node = GRAPH["synthesis_interpretation"]
    slide_cache = _slide_cache(tmp_path)
    retriever = MagicMock()
    retriever.retrieve.return_value = [_FakeRetrieved(patch_image=_FakePatchImage())]

    bundle = provider.for_node(
        node, slide_cache, query="q", retriever=retriever
    )

    assert bundle.thumbnail_path is not None
    assert len(bundle.patch_paths) == 1
    assert bundle.metadata["visual_policy"] == "both"


def test_graph_guided_navigator_patch_retrieve_routes_by_policy(tmp_path: Path):
    navigator = GraphGuidedNavigator(
        visual_method="patch_retrieve",
        cache_root=tmp_path,
    )
    slide_cache = _slide_cache(tmp_path)

    thumb_bundle = navigator.select_visual_bundle(
        GRAPH["compartment"],
        slide_cache,
        [],
        query="q",
        retriever=MagicMock(),
    )
    assert thumb_bundle.thumbnail_path is not None
    assert thumb_bundle.patch_paths == []

    retriever = MagicMock()
    retriever.retrieve.return_value = [_FakeRetrieved(patch_image=_FakePatchImage())]
    patch_bundle = navigator.select_visual_bundle(
        GRAPH["endometrium_assessment"],
        slide_cache,
        [],
        query="q",
        retriever=retriever,
    )
    assert patch_bundle.thumbnail_path is None
    assert patch_bundle.patch_paths


def test_thumbnail_ablation_ignores_graph_policy(tmp_path: Path):
    navigator = GraphGuidedNavigator(visual_method="thumbnail", cache_root=tmp_path)
    slide_cache = _slide_cache(tmp_path)
    retriever = MagicMock()

    for node_id in (
        "organ_procedure",
        "endometrium_assessment",
        "synthesis_interpretation",
    ):
        bundle = navigator.select_visual_bundle(
            GRAPH[node_id],
            slide_cache,
            [],
            query="q",
            retriever=retriever,
        )
        assert bundle.thumbnail_path is not None
        assert bundle.patch_paths == []

    retriever.retrieve.assert_not_called()
