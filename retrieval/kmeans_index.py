"""K-means candidate pool (K=100) — sole pre-filter for retrieval (no ABMIL)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def build_kmeans_index(
    embeddings: np.ndarray,
    k: int = 100,
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit MiniBatchKMeans and pick per-cluster representative patch indices.

    Returns
    -------
    centroid_indices : (k,) int — row indices into embeddings
    cluster_labels : (N,) int — cluster id per patch
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    n = emb.shape[0]
    if n == 0:
        raise ValueError("Cannot build K-means index from zero embeddings")
    k_eff = min(int(k), n)

    km = MiniBatchKMeans(n_clusters=k_eff, random_state=random_state, batch_size=256)
    labels = km.fit_predict(emb)
    centroids = km.cluster_centers_.astype(np.float32)

    centroid_indices = np.empty(k_eff, dtype=np.int64)
    for c in range(k_eff):
        mask = labels == c
        if not np.any(mask):
            centroid_indices[c] = 0
            continue
        members = emb[mask]
        dists = np.sum((members - centroids[c]) ** 2, axis=1)
        local_argmin = int(np.argmin(dists))
        centroid_indices[c] = int(np.flatnonzero(mask)[local_argmin])

    return centroid_indices, labels.astype(np.int64)
