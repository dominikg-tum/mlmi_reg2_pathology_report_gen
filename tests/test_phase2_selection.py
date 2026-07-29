"""Phase 2 reuses the stored pick and uses the LLM selector for legacy runs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from extraction.case_ids import physical_run_dir
from scripts.inference.run_phase2 import _load_selected_case_chain

CASE_KEY = "a.svs,b.svs"
SLIDES = ["a.svs", "b.svs"]


def _write_per_slide_chains(runs_dir: Path) -> None:
    for sid, category in zip(SLIDES, ["benign", "benign"], strict=True):
        phys = physical_run_dir(runs_dir, CASE_KEY, sid)
        phys.mkdir(parents=True)
        (phys / "cot_chain.json").write_text(
            json.dumps(
                {
                    "slide_id": sid,
                    "chain-of-thought": [
                        {
                            "node_id": "diagnosis",
                            "question": "Final category?",
                            "answer": category,
                            "next_question": "report",
                        }
                    ],
                    "node_path": ["diagnosis"],
                }
            )
            + "\n"
        )


def _llm_backend(chosen: str) -> SimpleNamespace:
    payload = json.dumps({"chosen_slide_id": chosen, "rationale": "Main finding."})
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
    )
    return SimpleNamespace(
        client=SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: response)
            )
        ),
        model="test-model",
    )


def test_legacy_case_run_is_migrated_with_llm_selector(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    _write_per_slide_chains(runs_dir)
    case_dir = runs_dir / CASE_KEY
    case_dir.mkdir(parents=True, exist_ok=True)
    # Legacy concatenated output from the previous merge implementation.
    (case_dir / "cot_chain.json").write_text(
        json.dumps({"slide_id": CASE_KEY, "fusion": "ss_llm", "chain-of-thought": []})
        + "\n"
    )

    chain, selected = _load_selected_case_chain(
        runs_dir,
        CASE_KEY,
        SLIDES,
        selector_backend=_llm_backend("b.svs"),
    )

    assert selected == "b.svs"
    assert chain["fusion"] == "ss_llm_pick"
    assert chain["selection_method"] == "llm"
    meta = json.loads((case_dir / "case_meta.json").read_text())
    assert meta["chosen_slide_id"] == "b.svs"


def test_stored_pick_is_reused_and_meta_rebuilt(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    _write_per_slide_chains(runs_dir)
    case_dir = runs_dir / CASE_KEY
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "cot_chain.json").write_text(
        json.dumps(
            {
                "slide_id": CASE_KEY,
                "physical_slides": SLIDES,
                "selected_slide_id": "b.svs",
                "selection_rationale": "Stored rationale.",
                "selection_method": "llm",
                "fusion": "ss_llm_pick",
                "chain-of-thought": [],
            }
        )
        + "\n"
    )

    # A selector that would pick differently must not be consulted.
    chain, selected = _load_selected_case_chain(
        runs_dir,
        CASE_KEY,
        SLIDES,
        selector_backend=_llm_backend("a.svs"),
    )

    assert selected == "b.svs"
    assert chain["selection_rationale"] == "Stored rationale."
    meta = json.loads((case_dir / "case_meta.json").read_text())
    assert meta["chosen_slide_id"] == "b.svs"
    assert meta["selection_method"] == "llm"
