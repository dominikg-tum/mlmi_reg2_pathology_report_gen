from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent.node_react import run_node_react
from graph.schema import InteractionType, Node, NodeKind, Tier, VisualPolicy, ZoomLevel
from retrieval.titan_cosine import RetrievedPatch
from vision.cache import SlideCache
from vision.mag_config import clamp_runtime_zoom


@dataclass
class _Backend:
    """Scripted JSON backend for Step A/B/C calls."""

    responses: list[dict]

    def complete_json(self, node, visual, *, system_prompt, user_prompt, guided_choice=None):
        out = self.responses.pop(0)
        return out, float(out.get("confidence", 1.0)), "raw"


class _Retriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, slide_cache, *, level="20x", exclude=None, **kwargs):
        self.calls.append({"query": query, "level": level, "exclude": set(exclude or [])})
        return [
            RetrievedPatch(
                patch_image=None,
                parent_image=None,
                level=level,
                coord=(1000, 2000),
                parent_coord=None,
                similarity=0.9,
                index=1,
            )
        ]


def _node() -> Node:
    return Node(
        id="n1",
        label="n1",
        question="Which compartment?",
        tier=Tier.LOCAL_FEATURES,
        node_kind=NodeKind.LOCAL,
        interaction=InteractionType.SINGLE_SELECT,
        description="",
        options=["endometrium", "myometrium"],
        edges={"endometrium": "n2", "myometrium": "n2"},
        zoom_level=ZoomLevel.X10,
        visual_policy=VisualPolicy.PATCH_RETRIEVE,
        requires_visual_evidence=True,
        is_leaf=False,
        root=False,
    )


def test_clamp_runtime_zoom_menu():
    assert clamp_runtime_zoom("40x") == "40x"
    assert clamp_runtime_zoom("5x") == "10x"
    assert clamp_runtime_zoom("1.25x") == "10x"
    assert clamp_runtime_zoom("nonsense") == "20x"


def test_node_react_retrieve_then_sufficient(tmp_path):
    node = _node()
    slide_cache = SlideCache(slide_id="case01.svs", cache_dir=tmp_path, thumbnail_path=None)
    retriever = _Retriever()

    backend = _Backend(
        responses=[
            {"answer_key": "endometrium", "rationale": "x", "confidence": 0.8},  # Step A
            {"sufficient": False, "missing_info": "need more"},  # Step B
            {"action": "retrieve", "sub_query": "different region", "zoom_level": "40x", "zoom_reason": ""},  # Step C
            {"answer_key": "endometrium", "rationale": "x", "confidence": 0.9},  # Step A again
            {"sufficient": True, "missing_info": ""},  # Step B again
        ]
    )

    result = run_node_react(
        node,
        backend=backend,
        retriever=retriever,
        slide_cache=slide_cache,
        wsi_path=None,
        prior_steps=[],
        max_iters=3,
    )

    assert result.answer_key == "endometrium"
    assert len(result.node_traces) == 2
    assert retriever.calls[0]["level"] == "20x"
    assert retriever.calls[1]["exclude"] == {1}

