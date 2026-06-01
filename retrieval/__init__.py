"""Patch retrieval (pluggable; TITAN cosine default for P2)."""

from retrieval.base import PatchRetriever, get_retriever

__all__ = ["PatchRetriever", "get_retriever", "TitanCosineRetriever", "SlideEmbeddings"]


def __getattr__(name: str):
    if name in ("TitanCosineRetriever", "SlideEmbeddings"):
        from retrieval.titan_cosine import SlideEmbeddings, TitanCosineRetriever

        if name == "SlideEmbeddings":
            return SlideEmbeddings
        return TitanCosineRetriever
    raise AttributeError(name)
