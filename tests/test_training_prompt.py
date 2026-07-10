"""CPU-only tests for the LoRA prompt builder (train/serve parity).

No torch / transformers / GPU needed — these only exercise the message assembly and its
agreement with the inference-time prompt helpers in ``agent.backends``.
"""

from __future__ import annotations

from agent.backends import (
    build_answer_prompt,
    system_prompt_for,
    visual_note_for_paths,
)
from graph import GRAPH, ROOT_ID
from graph.schema import NodeKind
from training.dataset import ChainSample
from training.prompt import build_chat_messages


def _sample(**over) -> ChainSample:
    base = dict(
        slide_id="TUM_Uterus_0001.svs",
        node_id="compartment",
        question=GRAPH["compartment"].question,
        target_answer="endometrium",
        visual_paths=["/tmp/img_0.jpg", "/tmp/img_1.jpg", "/tmp/img_2.jpg"],
        episodic_context="Q: What organ?\nA: uterus_curettage",
    )
    base.update(over)
    return ChainSample(**base)


def test_messages_structure_and_roles():
    sample = _sample()
    messages = build_chat_messages(sample)

    assert [m["role"] for m in messages] == ["system", "user", "assistant"]

    # user turn: one image part per visual path, then exactly one text part
    user = messages[1]["content"]
    image_parts = [p for p in user if p["type"] == "image"]
    text_parts = [p for p in user if p["type"] == "text"]
    assert len(image_parts) == len(sample.visual_paths)
    assert [p["image"] for p in image_parts] == sample.visual_paths
    assert len(text_parts) == 1

    # assistant turn carries the GT answer
    assert messages[2]["content"][0]["text"] == "endometrium"


def test_prompt_text_matches_inference_builder():
    """The training user-text must byte-match what the backend builds at serve time."""
    sample = _sample()
    node = GRAPH[sample.node_id]
    expected = build_answer_prompt(
        node,
        sample.episodic_context,
        visual_note_for_paths(sample.visual_paths),
    )

    messages = build_chat_messages(sample)
    user_text = [p for p in messages[1]["content"] if p["type"] == "text"][0]["text"]
    assert user_text == expected

    # system prompt also matches the inference selection for this node kind
    assert messages[0]["content"][0]["text"] == system_prompt_for(node)
    assert node.node_kind != NodeKind.REPORT


def test_include_target_false_omits_assistant():
    messages = build_chat_messages(_sample(), include_target=False)
    assert [m["role"] for m in messages] == ["system", "user"]


def test_no_visuals_note_and_no_image_parts():
    sample = _sample(node_id=ROOT_ID, question=GRAPH[ROOT_ID].question, visual_paths=[])
    messages = build_chat_messages(sample)
    user = messages[1]["content"]
    assert all(p["type"] == "text" for p in user)
    # empty visuals -> "none attached." branch in the prompt
    assert "none attached." in user[0]["text"]


def test_visual_note_counts_patches():
    assert visual_note_for_paths([]) == ""
    assert "whole-slide thumbnail attached." in visual_note_for_paths(["t.jpg"])
    note = visual_note_for_paths(["t.jpg", "p1.jpg", "p2.jpg"])
    assert "2 retrieved patch images" in note
