"""Flat episodic memory: append (Q,A) steps to the prompt."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryStep:
    node_id: str
    question: str
    answer: str


class FlatEpisodicMemory:
    def __init__(self) -> None:
        self._steps: list[MemoryStep] = []

    def append(self, node_id: str, question: str, answer: str) -> None:
        self._steps.append(MemoryStep(node_id, question, answer))

    def episodic_context(self) -> str:
        if not self._steps:
            return ""
        lines = [f"Q: {s.question}\nA: {s.answer}" for s in self._steps]
        return "\n".join(lines)

    def clear(self) -> None:
        self._steps.clear()
