import numpy as np

from retrieval.titan_cosine import TitanCosineRetriever
from vision.cache import SlideCache


class _FakeSlideCache(SlideCache):
    def __init__(self):
        super().__init__(slide_id="fake.svs", cache_dir=None)


def test_search_all_patches_uses_full_pool(tmp_path, monkeypatch):
    emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    coords = np.array([[0, 0], [100, 0], [0, 100]], dtype=np.int64)

    retriever = TitanCosineRetriever(
        text_encoder=lambda q: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        search_all_patches=True,
        d_min_20x=0,
    )

    slide_cache = _FakeSlideCache()

    def fake_load_embeddings(self, slide_cache, level):
        from retrieval.titan_cosine import SlideEmbeddings

        return SlideEmbeddings(embeddings=emb, coords=coords, patch_size_lv0=512)

    def fake_meta(self, slide_cache, level):
        return 512

    monkeypatch.setattr(retriever, "_load_embeddings", fake_load_embeddings.__get__(retriever, TitanCosineRetriever))
    monkeypatch.setattr(retriever, "_load_meta_patch_size", fake_meta.__get__(retriever, TitanCosineRetriever))

    results = retriever.retrieve(
        "test query",
        slide_cache,
        level="20x",
        k=1,
        return_images=False,
    )
    assert len(results) == 1
    assert results[0].index == 0
