import json
from pathlib import Path

from extraction.case_ids import load_cases_from_chains
from scripts.inference.run_baseline_batch import (
    BASELINES,
    default_runs_dir_for_baseline,
    load_slide_ids,
)


def test_load_slide_ids_filters_split(tmp_path: Path):
    chains = tmp_path / "chains.jsonl"
    rows = [
        {"slide_id": "a.svs", "split": "train", "extraction_status": "ok"},
        {"slide_id": "b.svs", "split": "test", "extraction_status": "ok"},
        {"slide_id": "c.svs", "split": "test", "extraction_status": "failed"},
    ]
    chains.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    assert load_slide_ids(chains, split="test") == ["b.svs"]
    assert load_slide_ids(chains, split="") == ["a.svs", "b.svs"]


def test_load_slide_ids_keeps_comma_case_key(tmp_path: Path):
    chains = tmp_path / "chains.jsonl"
    rows = [
        {
            "slide_id": "a.svs,b.svs",
            "split": "test",
            "extraction_status": "ok",
        },
    ]
    chains.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert load_slide_ids(chains, split="test") == ["a.svs,b.svs"]
    cases = load_cases_from_chains(chains, split="test")
    assert cases[0].physical_slides == ["a.svs", "b.svs"]


def test_baseline_specs():
    assert BASELINES["a"].memory == "flat"
    assert BASELINES["b1"].memory == "hipporag2"
    assert BASELINES["b2"].memory == "hybridrag"
    assert BASELINES["b2_cap"].memory == "hybridrag_cap"
    assert BASELINES["b2_cap"].name == "baseline_b2_hybridrag_cap"
    assert BASELINES["naive"].mode == "naive"
    assert BASELINES["a"].mode == "graph"


def test_default_runs_dir_for_baseline():
    path = default_runs_dir_for_baseline(
        BASELINES["a"],
        {"user": {"work_dir": "/tmp/work"}},
    )
    assert path == Path("/tmp/work/runs/baseline_a_flat")
