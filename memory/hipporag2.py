"""HippoRAG 2 semantic memory — NICK implements."""

from __future__ import annotations

from graph.schema import Node
from memory.base import SemanticMemory


class HippoRAG2Memory:
    def build_index(self, train_reports_path: str, *, split: str = "train") -> None:
        raise NotImplementedError(
            "NICK: build KG + PageRank index from train-split reports only. "
            "See HippoRAG 2 (ICML 2025) and MedMemoryBench."
        )

    def retrieve(self, node: Node, query: str, *, k: int = 5) -> str:
        raise NotImplementedError("NICK: retrieve diagnostic knowledge for current node.")
