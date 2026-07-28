"""run_case_phase1 with dummy backend writes SS-LLM case layout."""

from __future__ import annotations

import json
from pathlib import Path

import baselines.agent_runner as agent_runner
from baselines.agent_runner import AgentRunResult, run_case_phase1
from extraction.case_ids import CaseSpec, physical_run_dir


def test_run_case_phase1_writes_per_slide_and_merged(tmp_path, monkeypatch):
    case = CaseSpec(
        case_key="a.svs,b.svs",
        physical_slides=["a.svs", "b.svs"],
        split="test",
    )
    runs_dir = tmp_path / "runs"

    def _fake_traversal(**kwargs):
        sid = kwargs["slide_id"]
        chain = {
            "slide_id": sid,
            "chain-of-thought": [
                {
                    "node_id": f"n_{sid}",
                    "question": "Q?",
                    "answer": f"A-{sid}",
                    "next_question": "",
                }
            ],
            "node_path": [f"n_{sid}"],
            "report": "",
        }
        return AgentRunResult(steps=[], chain=chain, retrieval_log=[])

    monkeypatch.setattr(agent_runner, "run_agent_traversal", _fake_traversal)

    out = run_case_phase1(case, runs_dir=runs_dir, backend="dummy")
    assert out.exists()
    merged = json.loads(out.read_text())
    assert merged["slide_id"] == "a.svs,b.svs"
    assert merged["physical_slides"] == ["a.svs", "b.svs"]
    assert len(merged["chain-of-thought"]) == 2

    for sid in ("a.svs", "b.svs"):
        phys = physical_run_dir(runs_dir, case.case_key, sid) / "cot_chain.json"
        assert phys.exists()
        assert json.loads(phys.read_text())["slide_id"] == sid

    # skip_existing on case chain only when all physical chains exist
    out2 = run_case_phase1(case, runs_dir=runs_dir, backend="dummy", skip_existing=True)
    assert out2 == out

    # Stale case chain without a physical slide must re-run missing slides
    missing = physical_run_dir(runs_dir, case.case_key, "b.svs") / "cot_chain.json"
    missing.unlink()
    calls: list[str] = []

    def _tracking_traversal(**kwargs):
        calls.append(kwargs["slide_id"])
        return _fake_traversal(**kwargs)

    monkeypatch.setattr(agent_runner, "run_agent_traversal", _tracking_traversal)
    run_case_phase1(case, runs_dir=runs_dir, backend="dummy", skip_existing=True)
    assert calls == ["b.svs"]


def test_parse_naive_response():
    from baselines.direct_report import _parse_naive_response

    text = (
        "Q1: What tissue?\n"
        "A1: Endometrium\n"
        "Q2: Findings?\n"
        "A2: Benign\n"
        "FINAL: Benign endometrium."
    )
    chain = _parse_naive_response(text, "x.svs")
    assert chain["slide_id"] == "x.svs"
    assert chain["report"] == "Benign endometrium."
    assert len(chain["chain-of-thought"]) == 3
