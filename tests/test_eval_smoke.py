import json
import tempfile
from pathlib import Path

from eval.run_eval import load_jsonl, select_eval_keys
from eval.schemas import CaseRecord


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_eval_overlap():
    pred = {
        "slide_id": "s1",
        "chain-of-thought": [
            {"question": "Q1", "answer": "a1", "next_question": "Q2"},
            {"question": "Q2", "answer": "report text", "next_question": ""},
        ],
        "report": "report text",
    }
    gt = dict(pred)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pred.jsonl"
        g = Path(td) / "gt.jsonl"
        _write_jsonl(p, [pred])
        _write_jsonl(g, [gt])
        preds = load_jsonl(p)
        gts = load_jsonl(g)
        assert "s1" in preds
        assert CaseRecord.from_dict(pred).report == "report text"


def test_select_eval_keys_filters_by_gt_split():
    preds = {
        "train.svs": CaseRecord(slide_id="train.svs"),
        "test.svs": CaseRecord(slide_id="test.svs"),
    }
    gts = {
        "train.svs": CaseRecord(slide_id="train.svs", split="train"),
        "test.svs": CaseRecord(slide_id="test.svs", split="test"),
    }

    assert select_eval_keys(preds, gts) == ["test.svs", "train.svs"]
    assert select_eval_keys(preds, gts, split="test") == ["test.svs"]
    assert select_eval_keys(preds, gts, split="train") == ["train.svs"]
    assert select_eval_keys(preds, gts, split="val") == []
