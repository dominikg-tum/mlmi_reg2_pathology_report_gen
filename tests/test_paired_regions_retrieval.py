from __future__ import annotations

import numpy as np

from retrieval.titan_cosine import SlideEmbeddings, TitanCosineRetriever
from vision.cache import SlideCache


class _FakeSlideCache(SlideCache):
    def __init__(self):
        super().__init__(slide_id="fake.svs", cache_dir=None)


def _emb(coords, ps):
    n = len(coords)
    # make cosine scores identical, selection should be driven by distance filter
    return SlideEmbeddings(
        embeddings=np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (n, 1)),
        coords=np.asarray(coords, dtype=np.int64),
        patch_size_lv0=ps,
    )


def test_paired_regions_min_distance_filters_close_candidates(monkeypatch):
    retriever = TitanCosineRetriever(
        text_encoder=lambda q: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        search_all_patches=True,
        d_min_20x=0,
    )
    slide_cache = _FakeSlideCache()

    # two close coords and one far coord
    pools = {
        "20x": _emb([[0, 0], [256, 0], [4096, 0]], 512),
    }

    def fake_load(self, _slide_cache, level):
        return pools[level]

    def fake_meta(self, _slide_cache, level):
        return pools[level].patch_size_lv0

    monkeypatch.setattr(
        retriever, "_load_embeddings", fake_load.__get__(retriever, TitanCosineRetriever)
    )
    monkeypatch.setattr(
        retriever, "_load_meta_patch_size", fake_meta.__get__(retriever, TitanCosineRetriever)
    )

    res = retriever.retrieve(
        "q",
        slide_cache,
        level="20x",
        k=1,
        return_images=False,
        anchor_coord_lv0=(0, 0),
        min_dist_lv0_px=2048,
    )
    assert len(res) == 1
    assert res[0].coord == (4096, 0)

