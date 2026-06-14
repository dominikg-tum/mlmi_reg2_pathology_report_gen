"""Normalize VLM outputs into answers accepted by the execution graph."""

from __future__ import annotations

import json
import re

from graph.schema import InteractionType, Node


def _canonical_key(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    return re.sub(r"[^a-z0-9_]", "", text)


def _answer_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None

    if isinstance(decoded, dict):
        for key in ("answer", "choice", "label"):
            value = decoded.get(key)
            if isinstance(value, str):
                return value.strip()
    if isinstance(decoded, str):
        return decoded.strip()

    match = re.match(r"^(?:answer|choice|label)\s*:\s*(.+)$", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    return text.strip("`\"' \t\r\n.,;")


def normalize_answer(raw: str, node: Node) -> str | None:
    """Return a graph-compatible answer, or None when a choice cannot be resolved."""
    if node.interaction == InteractionType.FREE_TEXT:
        free_text = (raw or "").strip()
        return free_text or None

    text = _answer_text(raw)
    if node.interaction == InteractionType.MULTI_SELECT:
        return text or None

    if len(node.options) == 1:
        return node.options[0]
    if not text:
        return None
    if text in node.edges:
        return text

    raw_key = _canonical_key(text)
    exact = [option for option in node.options if _canonical_key(option) == raw_key]
    if len(exact) == 1:
        return exact[0]

    mentioned = [
        option
        for option in node.options
        if re.search(rf"(?<![a-z0-9]){re.escape(_canonical_key(option))}(?![a-z0-9])", raw_key)
    ]
    if len(mentioned) == 1:
        return mentioned[0]
    return None
