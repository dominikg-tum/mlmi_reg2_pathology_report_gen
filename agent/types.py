from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass
class Step:
    node_id: str
    question: str
    answer: str
    confidence: float
    next_question: str = ""
    node_traces: list[dict[str, Any]] = field(default_factory=list)
    # Which answer path produced this step: react | structured | plain.
    answer_branch: str = ""
    # When node_react/structured was requested but not used, why.
    answer_branch_skip_reason: str = ""
