"""Prompt pack for bounded per-node ReAct (Steps A/B/C)."""

from __future__ import annotations

import json

from graph.schema import Node


PERCEPTOR_PREAMBLE = (
    "You are an expert uterine pathology assistant. "
    "Use the attached whole-slide thumbnail as global context, and the retrieved tissue patches "
    "as local evidence. Do not invent findings that are not supported by the images."
)


STEP_A_SYSTEM = (
    "You are an expert uterine pathology assistant.\n"
    "Answer the question based only on the provided images and prior diagnostic answers.\n"
    "Output ONLY a JSON object with keys:\n"
    '{ "answer_key": "<one allowed key>", "rationale": "<brief evidence-based rationale>", "confidence": 0.0 }\n'
    "Confidence must be a number between 0 and 1."
)

STEP_B_SYSTEM = (
    "You are an expert uterine pathology assistant.\n"
    "Given a draft answer, judge whether the provided visual evidence is sufficient.\n"
    "Output ONLY a JSON object:\n"
    '{ "sufficient": true|false, "missing_info": "<what morphology is missing if insufficient>" }'
)

STEP_C_SYSTEM = (
    "You are an expert uterine pathology assistant.\n"
    "Evidence is insufficient. Choose one action to obtain missing morphology.\n"
    "Output ONLY a JSON object:\n"
    '{ "action": "retrieve"|"zoom", "sub_query": "<short noun phrase>", '
    '"zoom_level": "10x"|"20x"|"40x", "zoom_reason": "<brief reason>" }\n'
    "Use retrieve when a different region or feature is needed at similar magnification.\n"
    "Use zoom when the current region is correct but additional detail is needed."
)


def _prior_steps_text(prior_steps: list[tuple[str, str]] | None) -> str:
    if not prior_steps:
        return ""
    return "\n".join(f"{nid} -> {ans}" for nid, ans in prior_steps)


def format_step_a_user(
    *,
    node: Node,
    prior_steps: list[tuple[str, str]] | None = None,
    extra_context: str = "",
) -> str:
    allowed = "\n".join(f"- {k}" for k in node.options) if node.options else ""
    prior = _prior_steps_text(prior_steps)
    parts = [
        PERCEPTOR_PREAMBLE,
        f"Question:\n{node.question}",
    ]
    if node.description:
        parts.append(f"Diagnostic guidance:\n{node.description}")
    if allowed:
        parts.append(f"Allowed answer keys:\n{allowed}")
    if prior:
        parts.append(f"Prior diagnostic answers:\n{prior}")
    if extra_context.strip():
        parts.append(f"Retrieved knowledge:\n{extra_context.strip()}")
    return "\n\n".join(parts)


def format_step_b_user(
    *,
    node: Node,
    draft_json: dict,
    prior_steps: list[tuple[str, str]] | None = None,
) -> str:
    prior = _prior_steps_text(prior_steps)
    parts = [
        f"Question:\n{node.question}",
        f"Draft answer (Step A):\n{json.dumps(draft_json, ensure_ascii=False)}",
    ]
    if prior:
        parts.append(f"Prior diagnostic answers:\n{prior}")
    return "\n\n".join(parts)


def format_step_c_user(
    *,
    node: Node,
    missing_info: str,
    draft_json: dict,
) -> str:
    parts = [
        f"Question:\n{node.question}",
        f"Missing morphology (from reflect): {missing_info}",
        f"Draft answer: {json.dumps(draft_json, ensure_ascii=False)}",
        'Choose retrieve or zoom. If zoom, choose zoom_level from {"10x","20x","40x"}.',
    ]
    return "\n\n".join(parts)

