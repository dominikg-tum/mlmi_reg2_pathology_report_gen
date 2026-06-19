"""Tests for multi-slide case → primary WSI mapping."""

from __future__ import annotations

import json
from pathlib import Path

from data.case_slides import (
    canonical_wsi_id,
    iter_chain_records,
    parse_slide_ids,
    primary_wsi_for_baseline,
)


def test_parse_slide_ids_single():
    assert parse_slide_ids("abc.svs") == ["abc.svs"]


def test_parse_slide_ids_multi():
    raw = "a.svs,b.svs,c.svs"
    assert parse_slide_ids(raw) == ["a.svs", "b.svs", "c.svs"]


def test_primary_wsi_single():
    assert primary_wsi_for_baseline("only.svs") == "only.svs"


def test_primary_wsi_picks_corpus_index_by_default():
    assert primary_wsi_for_baseline("cervix.svs,corpus.svs") == "corpus.svs"


def test_primary_wsi_clamps_index():
    assert primary_wsi_for_baseline("a.svs,b.svs", index=5) == "b.svs"


def test_iter_chain_records_filters_split(tmp_path: Path):
    chains = tmp_path / "chains.jsonl"
    rows = [
        {"slide_id": "A.svs", "split": "train", "extraction_status": "ok"},
        {"slide_id": "B.svs", "split": "test", "extraction_status": "ok"},
        {"slide_id": "C.svs", "split": "test", "extraction_status": "failed"},
    ]
    chains.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    test_ids = [r["slide_id"] for r in iter_chain_records(chains, split="test")]
    assert test_ids == ["B.svs"]


def test_agent_run_uses_case_id_with_inference_wsi():
    from baselines.agent_runner import run_agent_traversal

    result = run_agent_traversal(
        backend="dummy",
        slide_id="a.svs,b.svs",
        wsi_slide_id="b.svs",
        skip_report_nodes=True,
    )
    assert result.chain["slide_id"] == "a.svs,b.svs"
    assert result.chain["inference_wsi"] == "b.svs"


def test_canonical_wsi_id_passes_through_tum():
    assert canonical_wsi_id("TUM_Uterus_0042.svs", wsi_map={}) == "TUM_Uterus_0042.svs"


def test_canonical_wsi_id_maps_uuid(tmp_path: Path):
    wsi_map = {"uuid-a.svs": "TUM_Uterus_0001.svs"}
    assert canonical_wsi_id("uuid-a.svs", wsi_map=wsi_map) == "TUM_Uterus_0001.svs"
    assert canonical_wsi_id("unknown.svs", wsi_map=wsi_map) == "unknown.svs"


def test_primary_wsi_resolves_uuid(tmp_path: Path):
    wsi_map = {
        "cervix.svs": "TUM_Uterus_0001.svs",
        "corpus.svs": "TUM_Uterus_0002.svs",
    }
    assert (
        primary_wsi_for_baseline("cervix.svs,corpus.svs", wsi_map=wsi_map)
        == "TUM_Uterus_0002.svs"
    )
