import numpy as np

from retrieval.titan_cosine import SlideEmbeddings, TitanCosineRetriever
from vision.cache import SlideCache


class _FakeSlideCache(SlideCache):
    def __init__(self):
        super().__init__(slide_id="fake.svs", cache_dir=None)


def _emb(coords, ps):
    n = len(coords)
    return SlideEmbeddings(
        embeddings=np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (n, 1)),
        coords=np.asarray(coords, dtype=np.int64),
        patch_size_lv0=ps,
    )


def test_retrieve_attaches_parent_for_all_adjacent_pairs(monkeypatch):
    retriever = TitanCosineRetriever(
        text_encoder=lambda q: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        search_all_patches=True,
        d_min_20x=0,
    )
    slide_cache = _FakeSlideCache()

    pools = {
        "10x": _emb([[0, 0]], 1024),
        "20x": _emb([[256, 256]], 512),
    }

    def fake_load(self, slide_cache, level):
        return pools[level]

    def fake_meta(self, slide_cache, level):
        return pools[level].patch_size_lv0

    monkeypatch.setattr(
        retriever, "_load_embeddings", fake_load.__get__(retriever, TitanCosineRetriever)
    )
    monkeypatch.setattr(
        retriever, "_load_meta_patch_size", fake_meta.__get__(retriever, TitanCosineRetriever)
    )

    results = retriever.retrieve(
        "test",
        slide_cache,
        level="20x",
        k=1,
        return_images=False,
    )
    assert len(results) == 1
    assert results[0].parent_level is None
    assert results[0].parent_coord is None
    assert results[0].grandparent_level is None


def test_retrieve_grandparent_for_integration_nodes(monkeypatch):
    retriever = TitanCosineRetriever(
        text_encoder=lambda q: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        search_all_patches=True,
        d_min_20x=0,
    )
    slide_cache = _FakeSlideCache()

    pools = {
        "5x": _emb([[0, 0]], 2048),
        "10x": _emb([[0, 0]], 1024),
        "20x": _emb([[256, 256]], 512),
    }

    def fake_load(self, slide_cache, level):
        return pools[level]

    def fake_meta(self, slide_cache, level):
        return pools[level].patch_size_lv0

    monkeypatch.setattr(
        retriever, "_load_embeddings", fake_load.__get__(retriever, TitanCosineRetriever)
    )
    monkeypatch.setattr(
        retriever, "_load_meta_patch_size", fake_meta.__get__(retriever, TitanCosineRetriever)
    )

    results = retriever.retrieve(
        "test",
        slide_cache,
        level="20x",
        k=1,
        return_images=False,
        tier="integration",
        node_kind="report",
    )
    assert results[0].parent_level is None
    assert results[0].grandparent_level is None
    assert results[0].grandparent_coord is None
