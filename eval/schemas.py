from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChainStep:
    question: str
    answer: str
    next_question: str = ""
    node_id: str = ""


@dataclass
class CaseRecord:
    slide_id: str
    chain: list[ChainStep] = field(default_factory=list)
    report: str = ""
    node_path: list[str] = field(default_factory=list)
    split: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CaseRecord:
        cot = raw.get("chain-of-thought") or raw.get("qa_chain") or []
        steps = []
        path = []
        for item in cot:
            q = item.get("question") or item.get("q", "")
            a = item.get("answer") or item.get("a", "")
            nq = item.get("next_question", "")
            nid = item.get("node_id", "")
            steps.append(ChainStep(q, a, nq, nid))
            if nid:
                path.append(nid)
        return cls(
            slide_id=raw.get("slide_id", ""),
            chain=steps,
            report=raw.get("report") or raw.get("final_report", ""),
            node_path=path or raw.get("node_path", []),
            split=str(raw.get("split", "") or ""),
        )
