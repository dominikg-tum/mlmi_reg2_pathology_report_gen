"""K-means centroid index (CPU only)."""

from __future__ import annotations

import numpy as np

from retrieval.kmeans_index import build_kmeans_index


def test_build_kmeans_index_shape():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((50, 8)).astype(np.float32)
    indices, labels = build_kmeans_index(emb, k=10)
    assert indices.shape == (10,)
    assert labels.shape == (50,)
    assert len(np.unique(indices)) == 10
