"""Turn a ``ChainSample`` into Qwen3-VL chat messages for LoRA fine-tuning.

Train/serve parity is the whole point of this module: the ``system`` prompt, the
``Visual evidence`` note, the episodic Q/A history, and the question layout are produced
by the exact same helpers the inference backends use (:mod:`agent.backends`). The only
difference is that here the assistant turn is filled with the ground-truth answer so the
model can be supervised on it.

The message format is the standard multimodal chat-template layout consumed by
``transformers`` processors (``apply_chat_template`` + ``process_vision_info``): a list
of ``{"role", "content"}`` dicts where ``content`` is a list of ``{"type": "image"|"text"}``
parts. Image parts carry the on-disk path under the ``"image"`` key so callers can load
pixels lazily.
"""

from __future__ import annotations

from typing import Any

from agent.backends import (
    build_answer_prompt,
    system_prompt_for,
    visual_note_for_paths,
)
from graph import GRAPH, Node
from graph.schema import (
    InteractionType,
    NodeKind,
    Tier,
    ZoomLevel,
)
from training.dataset import ChainSample

# Fallback used only if a sample references a node id absent from the current GRAPH
# (e.g. graph edited after the dataset was built). Keeps training robust instead of
# crashing on a single stale row.
_FALLBACK_NODE_KWARGS = dict(
    tier=Tier.LOCAL_FEATURES,
    node_kind=NodeKind.LOCAL,
    interaction=InteractionType.FREE_TEXT,
    zoom_level=ZoomLevel.X20,
)


def _node_for_sample(sample: ChainSample) -> Node:
    node = GRAPH.get(sample.node_id)
    if node is not None:
        return node
    return Node(
        id=sample.node_id,
        label=sample.node_id,
        question=sample.question,
        **_FALLBACK_NODE_KWARGS,
    )


def build_chat_messages(
    sample: ChainSample,
    *,
    include_target: bool = True,
) -> list[dict[str, Any]]:
    """Build system/user(/assistant) chat messages for one training sample.

    When ``include_target`` is False the assistant turn is omitted (useful for building
    a generation prompt for eval). Image parts reference ``sample.visual_paths`` in order
    (thumbnail first, then retrieved patches), matching inference selection.
    """
    node = _node_for_sample(sample)
    visual_note = visual_note_for_paths(sample.visual_paths)
    prompt = build_answer_prompt(node, sample.episodic_context, visual_note)

    user_content: list[dict[str, Any]] = [
        {"type": "image", "image": path} for path in sample.visual_paths
    ]
    user_content.append({"type": "text", "text": prompt})

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt_for(node)}]},
        {"role": "user", "content": user_content},
    ]
    if include_target:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample.target_answer}],
            }
        )
    return messages


def sample_image_paths(sample: ChainSample) -> list[str]:
    """Ordered image paths for a sample (parity with the chat-message image parts)."""
    return list(sample.visual_paths)
