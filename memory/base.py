"""Semantic memory factory for RAG ablations (NICK)."""

from __future__ import annotations

from typing import Protocol

from graph.schema import Node


class SemanticMemory(Protocol):
    def build_index(self, train_reports_path: str, *, split: str = "train") -> None: ...
    def retrieve(self, node: Node, query: str, *, k: int = 5) -> str: ...


def get_semantic_memory(method: str) -> SemanticMemory | None:
    if method in ("flat", "none", ""):
        return None
    if method == "hipporag2":
        from memory.hipporag2 import HippoRAG2Memory

        return HippoRAG2Memory()
    if method == "graphrag":
        from memory.graphrag import GraphRAGMemory

        return GraphRAGMemory()
    raise ValueError(f"Unknown memory method: {method!r}")
