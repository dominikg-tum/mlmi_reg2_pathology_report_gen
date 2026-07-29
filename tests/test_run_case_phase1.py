"""run_case_phase1 with dummy backend writes SS-LLM Pick case layout."""

from __future__ import annotations

import json
from pathlib import Path

import baselines.agent_runner as agent_runner
from baselines.agent_runner import AgentRunResult, run_case_phase1
from extraction.case_ids import CaseSpec, physical_run_dir


def test_run_case_phase1_writes_per_slide_and_selected_case(tmp_path, monkeypatch):
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
    selected = json.loads(out.read_text())
    assert selected["slide_id"] == "a.svs,b.svs"
    assert selected["physical_slides"] == ["a.svs", "b.svs"]
    assert selected["selected_slide_id"] == "a.svs"
    assert selected["fusion"] == "ss_llm_pick"
    assert len(selected["chain-of-thought"]) == 1
    assert selected["chain-of-thought"][0]["answer"] == "A-a.svs"

    meta = json.loads((out.parent / "case_meta.json").read_text())
    assert meta["chosen_slide_id"] == "a.svs"
    assert meta["selection_method"] == "severity_fallback"

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


def test_missing_case_meta_is_rebuilt_without_reselection(tmp_path, monkeypatch):
    case = CaseSpec(
        case_key="a.svs,b.svs",
        physical_slides=["a.svs", "b.svs"],
        split="test",
    )
    runs_dir = tmp_path / "runs"
    for sid in case.physical_slides:
        phys = physical_run_dir(runs_dir, case.case_key, sid)
        phys.mkdir(parents=True)
        (phys / "cot_chain.json").write_text(
            json.dumps({"slide_id": sid, "chain-of-thought": []}) + "\n"
        )

    case_dir = runs_dir / case.case_key
    (case_dir / "cot_chain.json").write_text(
        json.dumps(
            {
                "slide_id": case.case_key,
                "physical_slides": case.physical_slides,
                "selected_slide_id": "b.svs",
                "selection_rationale": "Malignant on the second slide.",
                "selection_method": "llm",
                "fusion": "ss_llm_pick",
                "chain-of-thought": [],
            }
        )
        + "\n"
    )

    def _fail(**kwargs):
        raise AssertionError("selection must not re-run when a valid pick is stored")

    monkeypatch.setattr(agent_runner, "select_slide_chain", _fail)

    out = run_case_phase1(case, runs_dir=runs_dir, backend="dummy", skip_existing=True)
    meta = json.loads((out.parent / "case_meta.json").read_text())
    assert meta["chosen_slide_id"] == "b.svs"
    assert meta["selection_method"] == "llm"
    assert json.loads(out.read_text())["selected_slide_id"] == "b.svs"


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


def test_run_case_phase1_skips_uncached_patch_slides(tmp_path, monkeypatch):
    """p0-p3: missing patch_embeddings skip the slide, keep runnable siblings."""
    case = CaseSpec(
        case_key="a.svs,b.svs,c.svs",
        physical_slides=["a.svs", "b.svs", "c.svs"],
        split="test",
    )
    runs_dir = tmp_path / "runs"
    cache_root = tmp_path / "cache"
    # Only b.svs has offline CONCH embeddings.
    emb = cache_root / "b.svs" / "patch_embeddings_20x.pt"
    emb.parent.mkdir(parents=True)
    emb.write_bytes(b"fake")

    called: list[str] = []

    def _fake_traversal(**kwargs):
        sid = kwargs["slide_id"]
        called.append(sid)
        chain = {
            "slide_id": sid,
            "chain-of-thought": [
                {
                    "node_id": "n",
                    "question": "Q?",
                    "answer": f"A-{sid}",
                    "next_question": "",
                }
            ],
            "node_path": ["n"],
            "report": "",
        }
        return AgentRunResult(steps=[], chain=chain, retrieval_log=[])

    monkeypatch.setattr(agent_runner, "run_agent_traversal", _fake_traversal)
    monkeypatch.setattr(agent_runner, "fixed_retrieval_pool", lambda: "20x")

    out = run_case_phase1(
        case,
        runs_dir=runs_dir,
        backend="dummy",
        visual="patch_retrieve",
        cache_root=cache_root,
    )
    assert called == ["b.svs"]
    meta = json.loads((out.parent / "case_meta.json").read_text())
    assert meta["chosen_slide_id"] == "b.svs"
    skipped = {row["slide_id"]: row["reason"] for row in meta["skipped_slides"]}
    assert skipped == {"a.svs": "no_patch_cache", "c.svs": "no_patch_cache"}


def test_run_case_phase1_errors_when_all_patch_caches_missing(tmp_path):
    case = CaseSpec(
        case_key="a.svs,b.svs",
        physical_slides=["a.svs", "b.svs"],
        split="test",
    )
    try:
        run_case_phase1(
            case,
            runs_dir=tmp_path / "runs",
            backend="dummy",
            visual="patch_retrieve",
            cache_root=tmp_path / "empty_cache",
        )
    except FileNotFoundError as exc:
        assert "No patch_embeddings cache" in str(exc)
        return
    raise AssertionError("expected FileNotFoundError")
