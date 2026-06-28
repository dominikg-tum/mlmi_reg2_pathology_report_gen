"""CPU-only smoke tests for the LoRA training-data builder.

Uses an injected fake visual provider so no TITAN / openslide / WSI files are needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from graph import GRAPH, ROOT_ID
from graph.schema import NodeKind
from training.dataset import ChainSample, build_training_jsonl, load_chain_samples
from vision.backends import VisualBundle


class _FakeProvider:
    """Returns a fixed thumbnail bundle regardless of node (parity not needed for test)."""

    def __init__(self, image_path: Path):
        self._image_path = image_path

    def for_node(self, node, slide_cache, *, query, retriever):
        return VisualBundle(
            thumbnail_path=self._image_path, metadata={"visual": "thumbnail"}
        )


def _make_image(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(120, 80, 80)).save(path, format="PNG")


def _report_node_id() -> str:
    for nid, node in GRAPH.items():
        if node.node_kind == NodeKind.REPORT:
            return nid
    raise AssertionError("graph has no report node")


def _write_chains(path: Path, report_id: str) -> None:
    records = [
        {
            "slide_id": "TUM_Uterus_0001.svs",
            "split": "train",
            "extraction_status": "ok",
            "chain-of-thought": [
                {"node_id": ROOT_ID, "question": GRAPH[ROOT_ID].question, "answer": "x"},
                {"node_id": report_id, "question": "Report?", "answer": "a report"},
            ],
        },
        {
            "slide_id": "TUM_Uterus_0002.svs",
            "split": "test",
            "extraction_status": "ok",
            "chain-of-thought": [
                {"node_id": ROOT_ID, "question": GRAPH[ROOT_ID].question, "answer": "y"},
            ],
        },
        {
            "slide_id": "TUM_Uterus_0003.svs",
            "split": "train",
            "extraction_status": "failed",
            "chain-of-thought": [],
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_build_training_jsonl_train_only(tmp_path: Path):
    img = tmp_path / "thumb.png"
    _make_image(img)
    chains = tmp_path / "chains.jsonl"
    out = tmp_path / "samples.jsonl"
    report_id = _report_node_id()
    _write_chains(chains, report_id)

    n = build_training_jsonl(
        chains,
        out,
        split="train",
        cache_root=None,
        image_root=tmp_path / "images",
        visual_provider=_FakeProvider(img),
        retriever=object(),
    )

    # Only the train slide's single non-report node should be written
    # (report node skipped, test slide filtered, failed slide skipped).
    assert n == 1
    samples = load_chain_samples(out)
    assert len(samples) == 1
    s = samples[0]
    assert isinstance(s, ChainSample)
    assert s.slide_id == "TUM_Uterus_0001.svs"
    assert s.node_id == ROOT_ID
    assert s.target_answer == "x"
    assert s.episodic_context == ""  # first node has no prior steps
    assert len(s.visual_paths) == 1
    assert Path(s.visual_paths[0]).exists()
    assert report_id not in {x.node_id for x in samples}


def test_episodic_context_accumulates(tmp_path: Path):
    """A second non-report node should carry the first node's Q/A as context."""
    img = tmp_path / "thumb.png"
    _make_image(img)

    # Find a non-report, non-root node to use as a second step.
    second_id = next(
        nid
        for nid, node in GRAPH.items()
        if node.node_kind != NodeKind.REPORT and nid != ROOT_ID
    )
    chains = tmp_path / "chains.jsonl"
    records = [
        {
            "slide_id": "S.svs",
            "split": "train",
            "extraction_status": "ok",
            "chain-of-thought": [
                {"node_id": ROOT_ID, "question": GRAPH[ROOT_ID].question, "answer": "x"},
                {"node_id": second_id, "question": GRAPH[second_id].question, "answer": "z"},
            ],
        }
    ]
    with chains.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    out = tmp_path / "samples.jsonl"
    build_training_jsonl(
        chains,
        out,
        cache_root=None,
        image_root=tmp_path / "images",
        visual_provider=_FakeProvider(img),
        retriever=object(),
    )
    samples = load_chain_samples(out)
    assert len(samples) == 2
    second = [s for s in samples if s.node_id == second_id][0]
    assert GRAPH[ROOT_ID].question in second.episodic_context
    assert "A: x" in second.episodic_context
