import json
from types import SimpleNamespace

from agent.slide_selector import (
    build_selected_case_chain,
    build_selection_prompt,
    fallback_selection,
    select_slide_chain,
    selection_from_case_chain,
)


def _chain(category: str, *, slide_id: str = "") -> dict:
    return {
        "slide_id": slide_id,
        "chain-of-thought": [
            {
                "node_id": "diagnosis",
                "question": "Final category?",
                "answer": category,
                "next_question": "report",
            }
        ],
        "node_path": ["diagnosis"],
        "report": f"{category} finding",
    }


def test_fallback_prefers_higher_diagnosis_severity():
    chains = [_chain("benign"), _chain("malignant"), _chain("premalignant")]
    selection = fallback_selection(chains, ["a.svs", "b.svs", "c.svs"])
    assert selection.chosen_slide_id == "b.svs"
    assert selection.method == "severity_fallback"


def test_fallback_tie_uses_first_slide():
    chains = [_chain("benign"), _chain("benign")]
    selection = fallback_selection(chains, ["a.svs", "b.svs"])
    assert selection.chosen_slide_id == "a.svs"


def test_llm_selection_parses_valid_json():
    payload = json.dumps(
        {
            "chosen_slide_id": "b.svs",
            "rationale": "Malignant diagnosis is most clinically significant.",
        }
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
    )
    completions = SimpleNamespace(create=lambda **_: response)
    backend = SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="test-model",
    )
    selection = select_slide_chain(
        [_chain("benign"), _chain("malignant")],
        ["a.svs", "b.svs"],
        backend=backend,
    )
    assert selection.chosen_slide_id == "b.svs"
    assert selection.method == "llm"


def test_invalid_llm_choice_uses_fallback():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"chosen_slide_id": "not-a-candidate.svs"}'
                )
            )
        ]
    )
    completions = SimpleNamespace(create=lambda **_: response)
    backend = SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="test-model",
    )
    selection = select_slide_chain(
        [_chain("malignant"), _chain("benign")],
        ["a.svs", "b.svs"],
        backend=backend,
    )
    assert selection.chosen_slide_id == "a.svs"
    assert selection.method == "severity_fallback"


def test_selected_case_chain_keeps_one_path_unchanged():
    chain = _chain("malignant", slide_id="b.svs")
    selection = fallback_selection([chain], ["b.svs"])
    selected = build_selected_case_chain(
        chain,
        case_key="a.svs,b.svs",
        physical_slides=["a.svs", "b.svs"],
        selection=selection,
    )
    assert selected["slide_id"] == "a.svs,b.svs"
    assert selected["selected_slide_id"] == "b.svs"
    assert selected["fusion"] == "ss_llm_pick"
    assert selected["chain-of-thought"] == chain["chain-of-thought"]


def test_selection_from_case_chain_restores_stored_pick():
    stored = {
        "fusion": "ss_llm_pick",
        "selected_slide_id": "b.svs",
        "selection_rationale": "Malignant on second slide.",
        "selection_method": "llm",
    }
    selection = selection_from_case_chain(stored, ["a.svs", "b.svs"])
    assert selection is not None
    assert selection.chosen_slide_id == "b.svs"
    assert selection.method == "llm"
    assert selection.rationale == "Malignant on second slide."


def test_selection_from_case_chain_rejects_stale_or_legacy():
    legacy = {"fusion": "ss_llm", "selected_slide_id": "b.svs"}
    assert selection_from_case_chain(legacy, ["a.svs", "b.svs"]) is None

    stale = {"fusion": "ss_llm_pick", "selected_slide_id": "gone.svs"}
    assert selection_from_case_chain(stale, ["a.svs", "b.svs"]) is None


def test_selection_prompt_contains_each_chain():
    prompt = build_selection_prompt(
        [_chain("benign"), _chain("malignant")],
        ["a.svs", "b.svs"],
    )
    assert "## Slide a.svs" in prompt
    assert "## Slide b.svs" in prompt
    assert "malignant" in prompt
