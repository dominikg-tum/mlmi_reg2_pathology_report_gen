"""Torch-free tests for the LoRA dataset builder (dataset.py)."""

from __future__ import annotations

import json
from pathlib import Path

from graph import GRAPH
from training.dataset import (
    ChainSample,
    build_chat_messages,
    build_training_jsonl,
    load_chain_samples,
    render_target,
)


def test_render_target_json_matches_structured_answer():
    raw = render_target("hysterectomy", answer_format="json")
    parsed = json.loads(raw)
    assert parsed["answer_key"] == "hysterectomy"
    assert "confidence" in parsed and "rationale" in parsed


def test_render_target_key():
    assert render_target("benign", answer_format="key") == "benign"


def test_build_chat_messages_image_placeholders():
    msgs = build_chat_messages("sys", "user", "target", n_images=3)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    image_parts = [c for c in msgs[1]["content"] if c["type"] == "image"]
    assert len(image_parts) == 3
    assert msgs[1]["content"][-1] == {"type": "text", "text": "user"}
    assert msgs[2]["content"][0]["text"] == "target"


def test_build_chat_messages_no_target_for_inference():
    msgs = build_chat_messages("sys", "user", None, n_images=0)
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_chain_sample_record_roundtrip():
    sample = ChainSample(
        slide_id="TUM_Uterus_0001.svs",
        node_id="n0",
        question="q?",
        target_answer="yes",
        visual_paths=["a.png", "b.png"],
        episodic_context="",
        system="sys",
        user="user",
        target=render_target("yes"),
        metadata={"n_images": 2},
    )
    back = ChainSample.from_record(sample.to_record())
    assert back == sample


def test_build_training_jsonl_from_chains(tmp_path: Path):
    # Pick a real choice node from the graph so options/prompt render correctly.
    choice_id = next(
        n.id
        for n in GRAPH.values()
        if n.interaction.value in ("single_select", "boolean")
        and n.options
        and not n.is_leaf
    )
    node = GRAPH[choice_id]
    gt = node.options[0]

    chains = tmp_path / "chains.jsonl"
    chains.write_text(
        json.dumps(
            {
                "slide_id": "TUM_Uterus_0001.svs",
                "split": "train",
                "extraction_status": "ok",
                "chain-of-thought": [
                    {"node_id": choice_id, "question": node.question, "answer": gt}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "samples.jsonl"
    # No retriever / slide_cache -> thumbnail-free, text-only samples (still valid).
    n = build_training_jsonl(chains, out, retriever=None, visual_method="none")
    assert n == 1

    samples = load_chain_samples(out)
    assert len(samples) == 1
    s = samples[0]
    assert s.node_id == choice_id
    assert s.target_answer == gt
    assert json.loads(s.target)["answer_key"] == gt
    assert node.question in s.user


def test_build_training_jsonl_skips_non_train_split(tmp_path: Path):
    chains = tmp_path / "chains.jsonl"
    chains.write_text(
        json.dumps(
            {
                "slide_id": "x.svs",
                "split": "test",
                "extraction_status": "ok",
                "chain-of-thought": [{"node_id": "any", "answer": "a"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "samples.jsonl"
    n = build_training_jsonl(chains, out, retriever=None, visual_method="none")
    assert n == 0
