"""GraphRAG semantic memory — NICK implements (ablation vs HippoRAG2)."""

from __future__ import annotations

from graph.schema import Node
from memory.base import SemanticMemory


class GraphRAGMemory:
    def build_index(self, train_reports_path: str, *, split: str = "train") -> None:
        raise NotImplementedError(
            "NICK: GraphRAG index over train reports — compare vs hipporag2 in ablations."
        )

    def retrieve(self, node: Node, query: str, *, k: int = 5) -> str:
        raise NotImplementedError("NICK: GraphRAG local/global search for node context.")
