"""Parse Phase 1/2 outputs into CaseRecord for REG² eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.schemas import CaseRecord, ChainStep
from graph import GRAPH


def chain_dict_to_record(
    chain: dict[str, Any],
    report: str = "",
) -> CaseRecord:
    """Build CaseRecord from cot_chain dict + optional Phase 2 report text."""
    slide_id = chain.get("slide_id", "")
    steps: list[ChainStep] = []
    node_path: list[str] = []

    for item in chain.get("chain-of-thought") or []:
        nid = item.get("node_id", "")
        steps.append(
            ChainStep(
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                next_question=item.get("next_question", ""),
                node_id=nid,
            )
        )
        if nid:
            node_path.append(nid)

    final_report = report or chain.get("report", "")
    return CaseRecord(
        slide_id=slide_id,
        chain=steps,
        report=final_report,
        node_path=node_path or chain.get("node_path", []),
    )


def record_to_eval_dict(record: CaseRecord) -> dict[str, Any]:
    return {
        "slide_id": record.slide_id,
        "chain-of-thought": [
            {
                "node_id": s.node_id,
                "question": s.question,
                "answer": s.answer,
                "next_question": s.next_question,
            }
            for s in record.chain
        ],
        "node_path": record.node_path,
        "report": record.report,
    }


def parse_slide_run(
    runs_dir: Path,
    slide_id: str,
    *,
    graph: dict | None = None,
) -> CaseRecord:
    """Load runs/{slide_id}/cot_chain.json + report.txt → CaseRecord."""
    _ = graph or GRAPH
    slide_dir = runs_dir / slide_id
    chain_path = slide_dir / "cot_chain.json"
    if not chain_path.exists():
        raise FileNotFoundError(f"Missing {chain_path}")

    chain = json.loads(chain_path.read_text())
    report_path = slide_dir / "report.txt"
    report = report_path.read_text() if report_path.exists() else ""
    return chain_dict_to_record(chain, report=report)


def write_pred_edges(record: CaseRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record_to_eval_dict(record)) + "\n")
