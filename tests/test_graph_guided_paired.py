"""GraphGuidedRetriever must forward paired_regions kwargs to the inner retriever."""

from __future__ import annotations

from retrieval.graph_guided import GraphGuidedRetriever
from retrieval.titan_cosine import RetrievedPatch
from vision.cache import SlideCache


class _Inner:
    def __init__(self):
        self.kwargs = None

    def retrieve(self, query, slide_cache, *, level="20x", k=3, exclude=None, **kwargs):
        self.kwargs = kwargs
        return [
            RetrievedPatch(
                patch_image=None,
                parent_image=None,
                level=level,
                coord=(0, 0),
                parent_coord=None,
                similarity=1.0,
                index=0,
            )
        ]


def test_graph_guided_forwards_paired_region_kwargs(tmp_path):
    inner = _Inner()
    retriever = GraphGuidedRetriever(inner)
    slide_cache = SlideCache(slide_id="s.svs", cache_dir=tmp_path, thumbnail_path=None)

    _ = retriever.retrieve(
        "q",
        slide_cache,
        level="20x",
        exclude={1},
        anchor_coord_lv0=(1000, 2000),
        min_dist_pool_px=2048,
        wsi_path=None,
        return_images=False,
    )

    assert inner.kwargs["anchor_coord_lv0"] == (1000, 2000)
    assert inner.kwargs["min_dist_pool_px"] == 2048
