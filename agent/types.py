from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Step:
    node_id: str
    question: str
    answer: str
    confidence: float
    next_question: str = ""
