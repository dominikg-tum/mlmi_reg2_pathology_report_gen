"""Deterministic traversal of the diagnostic graph.

The controller walks the graph. At each node it (1) builds a context-aware query,
(2) asks a retriever for patches, (3) asks an AnswerBackend for the answer, then
(4) routes to the next node using the node's hard-coded edges.

Swap the AnswerBackend to go from zero-shot -> fine-tuned without touching this
loop. Swap nothing else: navigation is identical at train and inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from graph.diagnostic_graph import GRAPH, ROOT_ID, Node


@dataclass
class Step:
    node_id: str
    question: str
    answer: str
    confidence: float


class Retriever(Protocol):
    def retrieve(self, query: str, slide, *, level: str, k: int = 3): ...


class AnswerBackend(Protocol):
    """Answers ONE node. Implementations: zero-shot Qwen, few-shot, LoRA-tuned VLM."""

    def answer(
        self, node: Node, patches, memory: list[Step]
    ) -> tuple[str, float]: ...


def build_query(node: Node, memory: list[Step]) -> str:
    """Context-aware query: prior answers sharpen retrieval for local nodes."""
    if not memory:
        return node.question
    context = "; ".join(f"{s.question} -> {s.answer}" for s in memory)
    return f"Given: {context}. {node.question}"


def route(node: Node, answer: str) -> str | None:
    """Deterministic edge selection. The model's answer only *indexes* the graph."""
    return node.next_id(answer)


def traverse(
    backend: AnswerBackend,
    retriever: Retriever | None = None,
    slide=None,
    graph: dict[str, Node] = GRAPH,
    root_id: str = ROOT_ID,
    max_steps: int = 64,
) -> list[Step]:
    """Walk root -> leaf. Returns the full reasoning chain (memory)."""
    node = graph[root_id]
    memory: list[Step] = []

    for _ in range(max_steps):
        query = build_query(node, memory)
        patches = (
            retriever.retrieve(query, slide, level=node.retrieval_level)
            if retriever is not None
            else None
        )
        answer, confidence = backend.answer(node, patches, memory)
        memory.append(Step(node.id, node.question, answer, confidence))

        next_id = route(node, answer)
        if next_id is None:  # leaf reached
            break
        node = graph[next_id]
    else:
        raise RuntimeError(f"traversal exceeded {max_steps} steps (cycle in graph?)")

    return memory


class DummyBackend:
    """Answers with each node's first option. Lets the loop run with no model."""

    def answer(self, node: Node, patches, memory: list[Step]) -> tuple[str, float]:
        if node.options:
            return node.options[0], 1.0
        if node.edges:
            return next(iter(node.edges)), 1.0
        return "<report>", 1.0


if __name__ == "__main__":
    chain = traverse(DummyBackend())
    for step in chain:
        print(f"{step.node_id:35s} {step.question}\n{'':35s} -> {step.answer}")
