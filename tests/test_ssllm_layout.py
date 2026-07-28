"""SS-LLM case layout + predictions join key = GT comma slide_id."""

from __future__ import annotations

import json
from pathlib import Path

from agent.report_writer import merge_case_chains, write_merged_case_chain
from eval.edge_parser import parse_slide_run, record_to_eval_dict
from eval.run_eval import load_jsonl, select_eval_keys
from extraction.case_ids import case_run_dir, physical_run_dir
from scripts.inference.build_predictions import iter_case_dirs


def test_ssllm_layout_predictions_match_gt_key(tmp_path: Path):
    case_key = "a.svs,b.svs"
    runs_dir = tmp_path / "runs"
    chains = [
        {
            "slide_id": "a.svs",
            "chain-of-thought": [
                {"node_id": "n1", "question": "Q?", "answer": "A", "next_question": ""},
            ],
            "node_path": ["n1"],
        },
        {
            "slide_id": "b.svs",
            "chain-of-thought": [
                {"node_id": "n2", "question": "Q2?", "answer": "B", "next_question": ""},
            ],
            "node_path": ["n2"],
        },
    ]
    for sid, chain in zip(["a.svs", "b.svs"], chains, strict=True):
        phys = physical_run_dir(runs_dir, case_key, sid)
        phys.mkdir(parents=True)
        (phys / "cot_chain.json").write_text(json.dumps(chain) + "\n")

    merged = merge_case_chains(chains, ["a.svs", "b.svs"], case_key)
    case_dir = case_run_dir(runs_dir, case_key)
    write_merged_case_chain(case_dir / "cot_chain.json", merged)
    (case_dir / "report.txt").write_text("Integrated CAP report.\n")

    # Nested slides/ must not appear as top-level case dirs.
    dirs = iter_case_dirs(runs_dir)
    assert [d.name for d in dirs] == [case_key]

    record = parse_slide_run(runs_dir, case_key)
    pred = record_to_eval_dict(record)
    assert pred["slide_id"] == case_key
    assert "Integrated CAP report" in pred["report"]

    pred_path = runs_dir / "predictions.jsonl"
    gt_path = runs_dir / "gt.jsonl"
    pred_path.write_text(json.dumps(pred) + "\n")
    gt_path.write_text(
        json.dumps(
            {
                "slide_id": case_key,
                "split": "test",
                "chain-of-thought": pred["chain-of-thought"],
                "report": "GT report",
            }
        )
        + "\n"
    )
    preds = load_jsonl(pred_path)
    gts = load_jsonl(gt_path)
    keys = select_eval_keys(preds, gts, split="test")
    assert keys == [case_key]
