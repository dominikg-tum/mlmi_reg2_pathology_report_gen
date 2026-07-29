from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent.node_react import run_node_react, tumor_anchor_from_steps
from agent.types import Step
from graph.schema import InteractionType, Node, NodeKind, Tier, VisualPolicy, ZoomLevel
from retrieval.titan_cosine import RetrievedPatch
from vision.cache import SlideCache
from vision.mag_config import clamp_runtime_zoom, node_react_max_iters


@dataclass
class _Backend:
    """Scripted JSON backend for Step A/B/C calls."""

    responses: list[dict]

    def complete_json(self, node, visual, *, system_prompt, user_prompt):
        out = self.responses.pop(0)
        return out, float(out.get("confidence", 1.0)), "raw"


class _Retriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, slide_cache, *, level="20x", exclude=None, **kwargs):
        self.calls.append(
            {
                "query": query,
                "level": level,
                "exclude": set(exclude or []),
                "kwargs": kwargs,
            }
        )
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


def test_node_react_paired_regions_passes_anchor_kwargs(tmp_path):
    node = _node()
    node.spatial_policy = "paired_regions"
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

    _ = run_node_react(
        node,
        backend=backend,
        retriever=retriever,
        slide_cache=slide_cache,
        wsi_path=None,
        prior_steps=[],
        max_iters=3,
        paired_regions=True,
    )

    # no tumor evidence in the chain → fall back to this node's own first hit
    assert "anchor_coord_lv0" not in retriever.calls[0]["kwargs"]
    assert retriever.calls[1]["kwargs"]["anchor_coord_lv0"] == (1000, 2000)
    assert retriever.calls[1]["kwargs"]["min_dist_pool_px"] > 0


def test_node_react_anchors_on_prior_tumor_step(tmp_path):
    """paired_regions must sample away from tumor coords found earlier in the chain."""
    node = _node()
    node.spatial_policy = "paired_regions"
    slide_cache = SlideCache(slide_id="case01.svs", cache_dir=tmp_path, thumbnail_path=None)
    retriever = _Retriever()

    tumor_step = Step(
        "mass_histologic_type",
        "Histologic type?",
        "epithelial_tumor",
        0.9,
        node_traces=[{"iter": 0, "patches": [{"coord": (50000, 60000), "similarity": 0.8}]}],
    )

    backend = _Backend(
        responses=[
            {"answer_key": "endometrium", "confidence": 0.9},
            {"sufficient": True, "missing_info": ""},
        ]
    )

    result = run_node_react(
        node,
        backend=backend,
        retriever=retriever,
        slide_cache=slide_cache,
        wsi_path=None,
        prior_steps=[tumor_step],
        max_iters=1,
        paired_regions=True,
    )

    # applied from the very first retrieve, anchored on the tumor coord
    assert retriever.calls[0]["kwargs"]["anchor_coord_lv0"] == (50000, 60000)
    assert result.node_traces[0]["paired_regions"]["anchor_source"] == "tumor_step"


def test_tumor_anchor_ignores_benign_smooth_muscle_answer():
    benign = Step(
        "smooth_muscle_tumor_assessment",
        "Smooth muscle tumor?",
        "none_benign",
        0.9,
        node_traces=[{"iter": 0, "patches": [{"coord": (10, 20)}]}],
    )
    assert tumor_anchor_from_steps([benign]) is None


def test_node_react_max_iters_defaults_to_config(tmp_path):
    node = _node()
    slide_cache = SlideCache(slide_id="case01.svs", cache_dir=tmp_path, thumbnail_path=None)
    retriever = _Retriever()
    budget = node_react_max_iters()

    # never sufficient → the loop must stop at the configured budget
    responses = []
    for _ in range(budget):
        responses.append({"answer_key": "endometrium", "confidence": 0.5})
        responses.append({"sufficient": False, "missing_info": "more"})
        responses.append({"action": "retrieve", "sub_query": "other region"})

    result = run_node_react(
        node,
        backend=_Backend(responses=responses),
        retriever=retriever,
        slide_cache=slide_cache,
        wsi_path=None,
        prior_steps=[],
    )

    assert len(retriever.calls) == budget
    assert len(result.node_traces) == budget


def test_node_react_step_a_carries_semantic_context(tmp_path):
    node = _node()
    slide_cache = SlideCache(slide_id="case01.svs", cache_dir=tmp_path, thumbnail_path=None)
    prompts_seen: list[str] = []

    class _PromptBackend(_Backend):
        def complete_json(self, node, visual, *, system_prompt, user_prompt):
            prompts_seen.append(user_prompt)
            return super().complete_json(
                node, visual, system_prompt=system_prompt, user_prompt=user_prompt
            )

    backend = _PromptBackend(
        responses=[
            {"answer_key": "endometrium", "confidence": 0.9},
            {"sufficient": True, "missing_info": ""},
        ]
    )

    _ = run_node_react(
        node,
        backend=backend,
        retriever=_Retriever(),
        slide_cache=slide_cache,
        wsi_path=None,
        prior_steps=[],
        max_iters=1,
        extra_context="CAP: report EIN when glands are crowded.",
    )

    assert "CAP: report EIN when glands are crowded." in prompts_seen[0]


def test_node_react_excludes_patches_after_insufficient_zoom(tmp_path, monkeypatch):
    """A zoom that does not resolve the node must not re-show the same patches."""
    from PIL import Image

    node = _node()
    slide_cache = SlideCache(slide_id="case01.svs", cache_dir=tmp_path, thumbnail_path=None)
    retriever = _Retriever()
    wsi_path = tmp_path / "case01.svs"
    wsi_path.write_bytes(b"fake")

    backend = _Backend(
        responses=[
            {"answer_key": "endometrium", "confidence": 0.6},  # A
            {"sufficient": False, "missing_info": "nuclear detail"},  # B
            {"action": "zoom", "zoom_level": "40x", "zoom_reason": "nuclei"},  # C
            {"answer_key": "endometrium", "confidence": 0.7},  # A post-zoom
            {"sufficient": False, "missing_info": "still unclear"},  # B post-zoom
            {"answer_key": "myometrium", "confidence": 0.9},  # A iter 2
            {"sufficient": True, "missing_info": ""},  # B iter 2
        ]
    )

    monkeypatch.setattr(
        "agent.node_react.zoom_crop_at_coord",
        lambda wsi, coord, *, from_zoom, to_zoom: Image.new("RGB", (32, 32)),
    )

    result = run_node_react(
        node,
        backend=backend,
        retriever=retriever,
        slide_cache=slide_cache,
        wsi_path=wsi_path,
        prior_steps=[],
        max_iters=2,
    )

    assert result.answer_key == "myometrium"
    assert len(retriever.calls) == 2
    assert retriever.calls[1]["exclude"] == {1}


def test_node_react_zoom_reanswers_with_zoom_patch(tmp_path, monkeypatch):
    """Zoom must attach a crop and re-run Step A/B before the next retrieve loop."""
    from PIL import Image

    node = _node()
    slide_cache = SlideCache(slide_id="case01.svs", cache_dir=tmp_path, thumbnail_path=None)
    retriever = _Retriever()
    wsi_path = tmp_path / "case01.svs"
    wsi_path.write_bytes(b"fake")

    seen_patch_counts: list[int] = []

    class _CountingBackend(_Backend):
        def complete_json(self, node, visual, *, system_prompt, user_prompt):
            seen_patch_counts.append(len(visual.patch_paths))
            return super().complete_json(
                node,
                visual,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

    backend = _CountingBackend(
        responses=[
            {"answer_key": "endometrium", "rationale": "x", "confidence": 0.7},  # A
            {"sufficient": False, "missing_info": "need nuclear detail"},  # B
            {"action": "zoom", "sub_query": "", "zoom_level": "40x", "zoom_reason": "nuclei"},  # C
            {"answer_key": "endometrium", "rationale": "zoom ok", "confidence": 0.95},  # A post-zoom
            {"sufficient": True, "missing_info": ""},  # B post-zoom
        ]
    )

    def _fake_zoom(wsi, coord, *, from_zoom, to_zoom):
        return Image.new("RGB", (64, 64), color=(200, 100, 50))

    monkeypatch.setattr("agent.node_react.zoom_crop_at_coord", _fake_zoom)

    result = run_node_react(
        node,
        backend=backend,
        retriever=retriever,
        slide_cache=slide_cache,
        wsi_path=wsi_path,
        prior_steps=[],
        max_iters=3,
    )

    assert result.answer_key == "endometrium"
    assert len(result.node_traces) == 1
    assert result.node_traces[0]["action"] == "zoom"
    assert "zoom_path" in result.node_traces[0]
    # A, B, then post-zoom A/B — last two calls must see the zoom crop
    assert seen_patch_counts[-2] >= 1
    assert seen_patch_counts[-1] >= 1
    assert len(retriever.calls) == 1  # no second retrieve after successful post-zoom

