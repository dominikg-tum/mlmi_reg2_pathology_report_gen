"""Bounded ReAct loop inside one graph node (Steps A/B/C)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent import prompts
from agent.answers import normalize_answer
from graph.schema import Node
from retrieval.base import PatchRetriever
from vision.backends import VisualBundle
from vision.cache import SlideCache
from vision.mag_config import (
    clamp_runtime_zoom,
    fixed_retrieval_pool,
    node_react_max_iters,
    paired_regions_config,
)
from vision.thumbnail import _bundle_from_retrieved
from vision.wsi_io import zoom_crop_at_coord


@dataclass
class NodeReactResult:
    answer_key: str
    confidence: float
    node_traces: list[dict[str, Any]]
    bundle: VisualBundle | None = None


# Nodes / answers that localize tumor bulk. paired_regions nodes must sample
# away from these coords (CAP/ISGyP: background endometrium and stage/extent are
# assessed off the mass).
_TUMOR_TOKENS = ("mass", "carcinoma", "tumor", "sarcoma", "malignan")
_NON_TUMOR_ANSWERS = (
    "none_benign",
    "none_of_above",
    "not_applicable",
    "unsure",
    "atypia_only",
)


def _steps_for_retrieval(prior_steps: list) -> list[tuple[str, str]]:
    return [(s.node_id, s.answer) for s in prior_steps]


def _step_localizes_tumor(step: Any) -> bool:
    answer = str(getattr(step, "answer", "") or "").lower()
    if answer in _NON_TUMOR_ANSWERS:
        return False
    node_id = str(getattr(step, "node_id", "") or "").lower()
    return any(t in node_id for t in _TUMOR_TOKENS) or any(
        t in answer for t in _TUMOR_TOKENS
    )


def tumor_anchor_from_steps(prior_steps: list) -> tuple[int, int] | None:
    """Top-ranked patch coord of the most recent tumor-localizing chain step.

    Requires that step to have been answered with ReAct, since only its traces
    carry retrieved coords. Returns None when the chain has no tumor evidence.
    """
    for step in reversed(prior_steps or []):
        if not _step_localizes_tumor(step):
            continue
        for trace in reversed(getattr(step, "node_traces", None) or []):
            patches = trace.get("patches") or []
            if not patches:
                continue
            coord = patches[0].get("coord")
            if coord is not None and len(coord) == 2:
                return (int(coord[0]), int(coord[1]))
    return None


def _best_coord(bundle: VisualBundle) -> tuple[int, int] | None:
    patches = bundle.metadata.get("retrieved_patches") or []
    if not patches:
        return None
    coord = patches[0].get("coord")  # sorted by similarity in retriever
    if coord is None or len(coord) != 2:
        return None
    return (int(coord[0]), int(coord[1]))


def _append_zoom_patch(
    bundle: VisualBundle,
    *,
    slide_cache: SlideCache,
    wsi_path: Path,
    coord_level0: tuple[int, int],
    zoom_level: str,
) -> Path:
    zoom_level = clamp_runtime_zoom(zoom_level)
    img = zoom_crop_at_coord(wsi_path, coord_level0, from_zoom="20x", to_zoom=zoom_level)
    out_dir = (slide_cache.cache_dir or Path(".")) / "retrieved_react"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"zoom_{zoom_level}_{coord_level0[0]}_{coord_level0[1]}.png"
    img.save(p)
    bundle.patch_paths.append(p)
    return p


def run_node_react(
    node: Node,
    *,
    backend: Any,
    retriever: PatchRetriever,
    slide_cache: SlideCache,
    wsi_path: Path | None,
    prior_steps: list,
    max_iters: int | None = None,
    paired_regions: bool = False,
    extra_context: str = "",
) -> NodeReactResult:
    pool = fixed_retrieval_pool()
    max_iters = node_react_max_iters() if max_iters is None else max_iters
    traces: list[dict[str, Any]] = []
    exclude: set[int] = set()
    anchor_coord: tuple[int, int] | None = None
    bundle: VisualBundle | None = None
    paired_cfg = paired_regions_config()
    paired_enabled = paired_regions and bool(paired_cfg.get("enabled", True))
    paired_min_dist = int(paired_cfg.get("min_dist_20x_px", 2048))
    paired_target = (
        paired_enabled
        and node.spatial_policy == "paired_regions"
        and paired_min_dist > 0
    )
    # Preferred anchor: tumor bulk found earlier in the chain. Falls back to this
    # node's own first hit when the chain carries no tumor coords.
    tumor_anchor = tumor_anchor_from_steps(prior_steps) if paired_target else None

    last_answer_key = ""
    last_conf = 0.0

    for it in range(max_iters):
        query = node.retrieval_text if it == 0 else node.retrieval_text_with_context(
            _steps_for_retrieval(prior_steps),
            sub_query=traces[-1].get("sub_query", ""),
        )

        anchor_kw = {}
        anchor_source = ""
        if paired_target:
            if tumor_anchor is not None:
                active_anchor, anchor_source = tumor_anchor, "tumor_step"
            elif it > 0 and anchor_coord is not None:
                active_anchor, anchor_source = anchor_coord, "self_iter0"
            else:
                active_anchor = None
            if active_anchor is not None:
                anchor_kw = {
                    "anchor_coord_lv0": active_anchor,
                    "min_dist_pool_px": paired_min_dist,
                }

        retrieved = retriever.retrieve(
            query,
            slide_cache,
            level=pool,
            exclude=exclude or None,
            **anchor_kw,
            wsi_path=wsi_path,
            return_images=wsi_path is not None,
            tier=node.tier.value,
            node_kind=node.node_kind.value,
        )
        bundle = _bundle_from_retrieved(retrieved, slide_cache, out_subdir="retrieved_react")
        if it == 0:
            anchor_coord = _best_coord(bundle)

        step_a_user = prompts.format_step_a_user(
            node=node,
            prior_steps=_steps_for_retrieval(prior_steps),
            extra_context=extra_context,
        )
        draft, conf_a, raw_a = backend.complete_json(
            node,
            bundle,
            system_prompt=prompts.STEP_A_SYSTEM,
            user_prompt=step_a_user,
        )
        answer_key = str(draft.get("answer_key", "")).strip()
        normalized = normalize_answer(answer_key, node)
        if normalized is not None:
            answer_key = normalized
        last_answer_key = answer_key
        last_conf = float(draft.get("confidence", conf_a) or conf_a)

        step_b_user = prompts.format_step_b_user(
            node=node,
            draft_json=draft,
            prior_steps=_steps_for_retrieval(prior_steps),
        )
        b, _conf_b, raw_b = backend.complete_json(
            node,
            bundle,
            system_prompt=prompts.STEP_B_SYSTEM,
            user_prompt=step_b_user,
        )
        sufficient = bool(b.get("sufficient", False))
        missing_info = str(b.get("missing_info", "")).strip()

        paired_active = bool(anchor_kw)
        trace: dict[str, Any] = {
            "iter": it,
            "pool": pool,
            "query": query,
            "patches": bundle.metadata.get("retrieved_patches") or [],
            "paired_regions": {
                "enabled": paired_target,
                "applied": paired_active,
                "anchor_source": anchor_source,
                "anchor_coord_lv0": anchor_kw.get("anchor_coord_lv0"),
                "min_dist_pool_px": paired_min_dist if paired_active else 0,
            },
            "draft": draft,
            "reflect": {"sufficient": sufficient, "missing_info": missing_info},
            "raw_step_a": raw_a,
            "raw_step_b": raw_b,
        }

        if sufficient:
            traces.append(trace)
            return NodeReactResult(
                answer_key=answer_key,
                confidence=last_conf,
                node_traces=traces,
                bundle=bundle,
            )

        step_c_user = prompts.format_step_c_user(
            node=node,
            missing_info=missing_info,
            draft_json=draft,
        )
        c, _conf_c, raw_c = backend.complete_json(
            node,
            bundle,
            system_prompt=prompts.STEP_C_SYSTEM,
            user_prompt=step_c_user,
        )
        action = str(c.get("action", "")).strip()
        sub_query = str(c.get("sub_query", "")).strip()
        zoom_level = str(c.get("zoom_level", "20x")).strip()
        zoom_reason = str(c.get("zoom_reason", "")).strip()

        trace.update(
            {
                "action": action,
                "sub_query": sub_query,
                "zoom_level": zoom_level,
                "zoom_reason": zoom_reason,
                "raw_step_c": raw_c,
            }
        )

        if action == "zoom" and wsi_path is not None:
            coord = _best_coord(bundle)
            if coord is not None:
                zoom_path = _append_zoom_patch(
                    bundle,
                    slide_cache=slide_cache,
                    wsi_path=wsi_path,
                    coord_level0=coord,
                    zoom_level=zoom_level,
                )
                # Re-run A/B with the zoom crop attached; otherwise the next loop
                # rebuilds the bundle from retrieval and the zoom evidence is lost.
                draft, conf_a, raw_a = backend.complete_json(
                    node,
                    bundle,
                    system_prompt=prompts.STEP_A_SYSTEM,
                    user_prompt=step_a_user,
                )
                answer_key = str(draft.get("answer_key", "")).strip()
                normalized = normalize_answer(answer_key, node)
                if normalized is not None:
                    answer_key = normalized
                last_answer_key = answer_key
                last_conf = float(draft.get("confidence", conf_a) or conf_a)

                b, _conf_b, raw_b = backend.complete_json(
                    node,
                    bundle,
                    system_prompt=prompts.STEP_B_SYSTEM,
                    user_prompt=prompts.format_step_b_user(
                        node=node,
                        draft_json=draft,
                        prior_steps=_steps_for_retrieval(prior_steps),
                    ),
                )
                sufficient = bool(b.get("sufficient", False))
                missing_info = str(b.get("missing_info", "")).strip()
                trace.update(
                    {
                        "zoom_path": str(zoom_path),
                        "post_zoom_draft": draft,
                        "post_zoom_reflect": {
                            "sufficient": sufficient,
                            "missing_info": missing_info,
                        },
                        "raw_step_a_post_zoom": raw_a,
                        "raw_step_b_post_zoom": raw_b,
                    }
                )
                if sufficient:
                    traces.append(trace)
                    return NodeReactResult(
                        answer_key=answer_key,
                        confidence=last_conf,
                        node_traces=traces,
                        bundle=bundle,
                    )

        # Still insufficient: exclude the patches already shown so the next
        # iteration ranks different regions. Also covers a zoom that could not
        # run (no WSI / no coord), which would otherwise re-retrieve the same
        # patches and burn the remaining iterations on identical evidence.
        for rp in retrieved:
            exclude.add(int(rp.index))

        traces.append(trace)

    return NodeReactResult(
        answer_key=last_answer_key,
        confidence=last_conf,
        node_traces=traces,
        bundle=bundle,
    )

