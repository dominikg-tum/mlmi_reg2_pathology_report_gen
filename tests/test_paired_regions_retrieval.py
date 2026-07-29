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


def _retriever_over_pool(monkeypatch, pools):
    retriever = TitanCosineRetriever(
        text_encoder=lambda q: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        search_all_patches=True,
        d_min_20x=0,
    )

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
    return retriever


def test_paired_regions_min_distance_filters_close_candidates(monkeypatch):
    slide_cache = _FakeSlideCache()

    # two close coords and one far coord
    pools = {
        "20x": _emb([[0, 0], [256, 0], [4096, 0]], 512),
    }
    retriever = _retriever_over_pool(monkeypatch, pools)

    res = retriever.retrieve(
        "q",
        slide_cache,
        level="20x",
        k=1,
        return_images=False,
        anchor_coord_lv0=(0, 0),
        min_dist_pool_px=2048,
    )
    assert len(res) == 1
    assert res[0].coord == (4096, 0)


def test_paired_regions_min_distance_scales_to_level0(monkeypatch):
    """On a 40x-base slide a 20x patch spans 1024 level-0 px, so the configured
    20x distance must double before it is compared against level-0 coords."""
    slide_cache = _FakeSlideCache()

    # patch_size_lv0=1024 for a 512 px 20x tile → 1 pool px = 2 level-0 px.
    pools = {
        "20x": _emb([[0, 0], [3072, 0], [8192, 0]], 1024),
    }
    retriever = _retriever_over_pool(monkeypatch, pools)

    res = retriever.retrieve(
        "q",
        slide_cache,
        level="20x",
        k=1,
        return_images=False,
        anchor_coord_lv0=(0, 0),
        min_dist_pool_px=2048,
    )
    # 3072 level-0 px is only 1536 px at 20x, so it must be rejected.
    assert len(res) == 1
    assert res[0].coord == (8192, 0)

