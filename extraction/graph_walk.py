"""Graph-aligned CoT extraction: simulate Phase 1 traversal over report text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from agent.types import Step
from graph.schema import InteractionType, Node, NodeKind

SYSTEM_PROMPT = (
    "You are a pathology expert. Answer the current diagnostic question based ONLY "
    "on the provided pathology report. Pick exactly one allowed option when choices "
    "are given. If the report does not support any option, answer 'not_mentioned'."
)

MAX_WALK_STEPS = 64


class GraphWalkError(Exception):
    """Raised when graph walk cannot produce a valid chain."""


@dataclass
class WalkResult:
    steps: list[Step]
    node_path: list[str]


class LlmClient(Protocol):
    def chat_completions_create(self, **kwargs: Any) -> Any: ...


def _canonical_key(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text


def build_step_prompt(node: Node, prior_steps: list[Step], report: str) -> str:
    """Mirror agent.controller.build_query with full report context."""
    if not prior_steps:
        question = node.question
    else:
        context = "; ".join(f"{s.question} -> {s.answer}" for s in prior_steps)
        question = f"Given: {context}. {node.question}"

    parts = [f"Pathology report:\n{report.strip()}", f"Current question: {question}"]
    if node.description:
        parts.append(f"Context: {node.description.strip()}")
    if node.options:
        opts = ", ".join(node.options)
        parts.append(f"Allowed answers (pick exactly one): {opts}")
    return "\n\n".join(parts)


def normalize_answer(raw: str, node: Node) -> str | None:
    """Map LLM output to a valid graph edge key, or None if no match."""
    raw = (raw or "").strip()
    if not raw:
        return None

    if len(node.options) == 1:
        return node.options[0]

    if raw in node.edges:
        return raw

    raw_key = _canonical_key(raw)
    for opt in node.options:
        if _canonical_key(opt) == raw_key:
            return opt
        if opt in raw or raw in opt:
            return opt

    for opt in node.options:
        if _canonical_key(opt) in raw_key or raw_key in _canonical_key(opt):
            return opt

    return None


def extract_node_answer(
    client: Any,
    model: str,
    node: Node,
    report: str,
    prior_steps: list[Step],
    *,
    retry: bool = True,
) -> str:
    """One vLLM call per node with guided_choice when options exist."""
    prompt = build_step_prompt(node, prior_steps, report)
    extra_body: dict[str, Any] = {}
    if node.interaction in (InteractionType.SINGLE_SELECT, InteractionType.BOOLEAN):
        if node.options:
            extra_body["guided_choice"] = node.options

    def _call(user_suffix: str = "") -> str:
        content = prompt + user_suffix
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            extra_body=extra_body or None,
        )
        return (response.choices[0].message.content or "").strip()

    raw = _call()
    answer = normalize_answer(raw, node)
    if answer is not None:
        return answer

    if retry:
        retry_suffix = (
            "\n\n[Retry: respond with exactly one allowed answer key from the list. "
            "Use not_mentioned only if none apply.]"
        )
        raw = _call(retry_suffix)
        answer = normalize_answer(raw, node)
        if answer is not None:
            return answer

    raise GraphWalkError(
        f"Invalid answer for node {node.id!r}: {raw!r}; valid options: {node.options}"
    )


def walk_graph(
    report: str,
    graph: dict[str, Node],
    root_id: str,
    client: Any,
    model: str,
) -> WalkResult:
    """Traverse graph node-by-node; stop before the report leaf (Phase 1 parity)."""
    if not report.strip():
        raise GraphWalkError("Empty report text")

    node = graph[root_id]
    steps: list[Step] = []
    seen: set[str] = set()

    for _ in range(MAX_WALK_STEPS):
        if node.id in seen:
            raise GraphWalkError(f"Cycle detected at node {node.id!r}")
        seen.add(node.id)

        if node.node_kind == NodeKind.REPORT:
            break

        answer = extract_node_answer(client, model, node, report, steps)
        next_id = node.next_id(answer)
        next_q = graph[next_id].question if next_id and next_id in graph else ""

        steps.append(
            Step(
                node_id=node.id,
                question=node.question,
                answer=answer,
                confidence=1.0,
                next_question=next_q,
            )
        )

        if next_id is None:
            break

        next_node = graph[next_id]
        if next_node.node_kind == NodeKind.REPORT:
            break
        node = next_node
    else:
        raise GraphWalkError(f"Traversal exceeded {MAX_WALK_STEPS} steps")

    return WalkResult(steps=steps, node_path=[s.node_id for s in steps])


def steps_to_chain_dict(
    slide_id: str,
    steps: list[Step],
    report: str,
    split: str,
    *,
    extraction_status: str = "ok",
    error: str = "",
) -> dict[str, Any]:
    """Build a chains.jsonl record matching eval/schemas.py."""
    record: dict[str, Any] = {
        "slide_id": slide_id,
        "split": split,
        "chain-of-thought": [
            {
                "node_id": s.node_id,
                "question": s.question,
                "answer": s.answer,
                "next_question": s.next_question,
            }
            for s in steps
        ],
        "node_path": [s.node_id for s in steps],
        "report": report,
        "extraction_status": extraction_status,
    }
    if error:
        record["error"] = error
    return record
