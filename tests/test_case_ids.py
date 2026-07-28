import json
from pathlib import Path

from extraction.case_ids import (
    CaseSpec,
    case_run_dir,
    case_spec_from_key,
    load_cases_from_chains,
    parse_slide_ids,
    physical_run_dir,
)


def test_parse_slide_ids_single_and_multi():
    assert parse_slide_ids("a.svs") == ["a.svs"]
    assert parse_slide_ids("a.svs, b.svs") == ["a.svs", "b.svs"]
    assert parse_slide_ids("  a.svs, ,b.svs  ") == ["a.svs", "b.svs"]
    assert parse_slide_ids("") == []


def test_case_spec_from_key():
    case = case_spec_from_key("a.svs,b.svs", split="test")
    assert case.case_key == "a.svs,b.svs"
    assert case.physical_slides == ["a.svs", "b.svs"]
    assert case.split == "test"


def test_run_dir_helpers(tmp_path: Path):
    case_key = "a.svs,b.svs"
    assert case_run_dir(tmp_path, case_key) == tmp_path / case_key
    assert physical_run_dir(tmp_path, case_key, "a.svs") == tmp_path / case_key / "slides" / "a.svs"


def test_load_cases_from_chains_multi_slide(tmp_path: Path):
    chains = tmp_path / "chains.jsonl"
    rows = [
        {
            "slide_id": "a.svs,b.svs",
            "split": "test",
            "extraction_status": "ok",
        },
        {
            "slide_id": "c.svs",
            "split": "train",
            "extraction_status": "ok",
        },
        {
            "slide_id": "d.svs",
            "split": "test",
            "extraction_status": "failed",
        },
    ]
    chains.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    cases = load_cases_from_chains(chains, split="test")
    assert len(cases) == 1
    assert cases[0] == CaseSpec(
        case_key="a.svs,b.svs",
        physical_slides=["a.svs", "b.svs"],
        split="test",
    )

    all_cases = load_cases_from_chains(chains, split="")
    assert [c.case_key for c in all_cases] == ["a.svs,b.svs", "c.svs"]
