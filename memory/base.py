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
        from pathlib import Path

        from memory.hipporag2 import HippoRAG2Memory

        repo = Path(__file__).resolve().parents[1]
        index_path = repo / "data" / "memory" / "hipporag_index.json"
        return HippoRAG2Memory(index_path=index_path)
    if method == "graphrag":
        from memory.graphrag import GraphRAGMemory

        return GraphRAGMemory()
    if method == "hybridrag":
        from memory.hybridrag import HybridRAGMemory

        return HybridRAGMemory()
    raise ValueError(f"Unknown memory method: {method!r}")
