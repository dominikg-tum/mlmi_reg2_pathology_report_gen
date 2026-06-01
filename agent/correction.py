"""Self-correction hooks (Group 2) — retry and consistency checks."""

from __future__ import annotations

from agent.types import Step
from graph.schema import Node


def should_retry(confidence: float, threshold: float = 0.65) -> bool:
    return confidence < threshold


def check_consistency(memory: list[Step], node: Node, answer: str) -> list[str]:
    """Rule-based flags; extend with graph-specific logic."""
    warnings: list[str] = []
    if not memory:
        return warnings
    # Example: flag contradictory invasion answers (placeholder)
    answers_lower = " ".join(s.answer.lower() for s in memory)
    if "no invasion" in answers_lower and "invasion" in answer.lower() and "no" not in answer.lower():
        warnings.append("Possible inconsistency: prior denial of invasion vs current answer")
    return warnings
