"""Graph store + case memory API (get_node, next, retrieve_context)."""

from __future__ import annotations

from typing import Protocol

from graph.schema import InteractionType, Node
from memory.base import SemanticMemory, get_semantic_memory
from memory.episodic import FlatEpisodicMemory


class GraphStore(Protocol):
    def get_node(self, node_id: str) -> Node: ...
    def next(self, node_id: str, answer: str) -> str | None: ...


class JsonGraphStore:
    def __init__(self, graph: dict[str, Node]):
        self._graph = graph

    def get_node(self, node_id: str) -> Node:
        return self._graph[node_id]

    def next(self, node_id: str, answer: str) -> str | None:
        return self._graph[node_id].next_id(answer)


class CaseMemory:
    """Episodic buffer + optional semantic RAG (NICK)."""

    def __init__(
        self,
        semantic: SemanticMemory | None = None,
        *,
        exclude_case_key: str | None = None,
    ) -> None:
        self.episodic = FlatEpisodicMemory()
        self.semantic = semantic
        self.exclude_case_key = (exclude_case_key or "").strip() or None

    def append(self, node_id: str, question: str, answer: str) -> None:
        self.episodic.append(node_id, question, answer)
        if self.semantic is not None and hasattr(self.semantic, "online_update"):
            try:
                self.semantic.online_update(node_id, question, answer)
            except Exception:
                pass

    def episodic_context(self) -> str:
        return self.episodic.episodic_context()

    def semantic_context(self, node: Node, query: str, *, k: int = 5) -> str:
        """Semantic RAG hits only. Empty for `flat` memory.

        Kept separate from `retrieve_context` so prompt paths that already
        serialize prior steps themselves (Step A/B/C) can add RAG knowledge
        without duplicating the episodic buffer.
        """
        if self.semantic is None:
            return ""
        try:
            return (
                self.semantic.retrieve(
                    node,
                    query,
                    k=k,
                    exclude_case_key=self.exclude_case_key,
                )
                or ""
            )
        except TypeError:
            # Older SemanticMemory stubs without exclude_case_key.
            try:
                return self.semantic.retrieve(node, query, k=k) or ""
            except NotImplementedError:
                return ""
        except NotImplementedError:
            return ""

    def retrieve_context(self, node: Node, query: str, *, k: int = 5) -> str:
        parts = []
        ep = self.episodic_context()
        if ep:
            parts.append(f"Prior steps:\n{ep}")
        sem = self.semantic_context(node, query, k=k)
        if sem:
            parts.append(f"Retrieved knowledge:\n{sem}")
        return "\n\n".join(parts)

    @classmethod
    def from_config(
        cls,
        memory_method: str = "flat",
        *,
        exclude_case_key: str | None = None,
    ) -> CaseMemory:
        return cls(
            semantic=get_semantic_memory(memory_method),
            exclude_case_key=exclude_case_key,
        )
