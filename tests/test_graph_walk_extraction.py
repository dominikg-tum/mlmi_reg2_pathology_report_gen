"""Tests for WP3 graph-walk ground-truth extraction."""

from __future__ import annotations

from dataclasses import dataclass

from agent.types import Step
from extraction.graph_walk import (
    GraphWalkError,
    build_step_prompt,
    extract_node_answer,
    normalize_answer,
    steps_to_chain_dict,
    walk_graph,
)
from extraction.labels_io import assign_splits, load_existing_slide_ids, write_chains_jsonl
from graph import GRAPH, load_graph


def _node(node_id: str):
    return GRAPH[node_id]


def test_normalize_answer_exact_and_case():
    node = _node("compartment")
    assert normalize_answer("endometrium", node) == "endometrium"
    assert normalize_answer("Endometrium", node) == "endometrium"
    assert normalize_answer(" endometrium ", node) == "endometrium"


def test_normalize_answer_single_option():
    node = _node("integration_synopsis")
    assert normalize_answer("anything", node) == "proceed_to_report"


def test_normalize_answer_invalid():
    node = _node("compartment")
    assert normalize_answer("totally_unknown_xyz", node) is None


def test_build_step_prompt_includes_prior_steps():
    node = _node("endometrium_adequacy")
    prior = [
        Step("organ_procedure", "Organ?", "uterus_hysterectomy", 1.0),
        Step("compartment", "Compartment?", "endometrium", 1.0),
    ]
    prompt = build_step_prompt(node, prior, "Sample report text.")
    assert "Organ? -> uterus_hysterectomy" in prompt
    assert "Sample report text." in prompt
    assert "endometrium_adequacy" not in prompt or node.question in prompt


def test_steps_to_chain_dict_schema():
    steps = [
        Step("organ_procedure", "Organ?", "uterus_hysterectomy", 1.0, "Compartment?"),
        Step("compartment", "Compartment?", "endometrium", 1.0, ""),
    ]
    record = steps_to_chain_dict("CASE.svs", steps, "Final report.", "train")
    assert record["slide_id"] == "CASE.svs"
    assert record["split"] == "train"
    assert record["report"] == "Final report."
    assert record["node_path"] == ["organ_procedure", "compartment"]
    assert record["chain-of-thought"][0]["node_id"] == "organ_procedure"
    assert record["extraction_status"] == "ok"


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]


class _FakeCompletions:
    def __init__(self, answers: list[str]):
        self._answers = list(answers)
        self.calls = 0

    def create(self, **kwargs):
        if self.calls >= len(self._answers):
            raise RuntimeError("no more mock answers")
        ans = self._answers[self.calls]
        self.calls += 1
        return _FakeResponse([_FakeChoice(_FakeMessage(ans))])


class _FakeClient:
    def __init__(self, answers: list[str]):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(answers)})()


def test_walk_graph_endometrium_prefix():
    """Mock answers along organ -> compartment -> adequate path."""
    graph, root_id = load_graph()
    answers = [
        "uterus_hysterectomy",
        "endometrium",
        "adequate",
        "malignant_pattern",
        "endometrioid",
        "grade_2",
        "present_deep",
        "proceed_to_report",
    ]
    client = _FakeClient(answers)
    result = walk_graph("Endometrioid carcinoma grade 2.", graph, root_id, client, "mock")
    assert result.node_path[0] == "organ_procedure"
    assert result.node_path[1] == "compartment"
    assert result.steps[1].answer == "endometrium"
    assert "report" not in result.node_path
    assert result.node_path[-1] == "integration_synopsis"


def test_extract_node_answer_retries_on_invalid():
    node = _node("compartment")
    client = _FakeClient(["garbage", "endometrium"])
    answer = extract_node_answer(client, "mock", node, "Report.", [], retry=True)
    assert answer == "endometrium"
    assert client.chat.completions.calls == 2


def test_extract_node_answer_raises_without_valid_retry():
    node = _node("compartment")
    client = _FakeClient(["garbage", "still_bad"])
    try:
        extract_node_answer(client, "mock", node, "Report.", [], retry=True)
    except GraphWalkError:
        return
    raise AssertionError("expected GraphWalkError")


def test_assign_splits():
    splits = assign_splits(100, test_n=30)
    assert len(splits) == 100
    assert sum(1 for v in splits.values() if v == "test") == 30


def test_jsonl_roundtrip():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chains.jsonl"
        write_chains_jsonl(
            [{"slide_id": "A.svs", "extraction_status": "ok", "chain-of-thought": []}],
            path,
        )
        assert load_existing_slide_ids(path) == {"A.svs"}
