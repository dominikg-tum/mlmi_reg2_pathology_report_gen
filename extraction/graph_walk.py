"""Graph-aligned CoT extraction: simulate Phase 1 traversal over report text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agent.answers import _answer_text, normalize_answer
from agent.types import Step
from graph.schema import InteractionType, Node, NodeKind

NOT_MENTIONED = "not_mentioned"
SYNTHESIS_NODE_ID = "synthesis_interpretation"
DIAGNOSIS_NODE_ID = "diagnosis"

SYSTEM_PROMPT = (
    "You are a pathology expert. Answer the current diagnostic question based ONLY "
    "on the provided pathology report. Pick exactly one allowed option when choices "
    "are given. Prefer the closest supported option when the report partially "
    "addresses the question. Use not_mentioned only when the report clearly does "
    "not address this question or the specimen is outside the uterine graph scope."
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


def normalize_extraction_answer(raw: str, node: Node) -> str | None:
    """Graph-walk answer normalization; accepts not_mentioned for WP3 extraction."""
    text = _answer_text(raw).lower().replace(" ", "_").replace("-", "_")
    if text in ("not_mentioned", "notmentioned", "n/a", "na", "none"):
        return NOT_MENTIONED
    return normalize_answer(raw, node)


def _guided_choices(node: Node) -> list[str]:
    if node.interaction not in (InteractionType.SINGLE_SELECT, InteractionType.BOOLEAN):
        return []
    if not node.options:
        return []
    choices = list(node.options)
    if NOT_MENTIONED not in choices:
        choices.append(NOT_MENTIONED)
    return choices


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
        opts = ", ".join(_guided_choices(node))
        parts.append(f"Allowed answers (pick exactly one): {opts}")
    return "\n\n".join(parts)


def _append_step(
    steps: list[Step],
    node: Node,
    answer: str,
    graph: dict[str, Node],
    next_id: str | None,
) -> None:
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


def _resolve_next_node(
    node: Node,
    answer: str,
    graph: dict[str, Node],
) -> tuple[str | None, Node | None]:
    """Return (next_id, next_node). Short-circuit to synthesis on not_mentioned."""
    if answer == NOT_MENTIONED:
        if node.id in (SYNTHESIS_NODE_ID, DIAGNOSIS_NODE_ID):
            return None, None
        if SYNTHESIS_NODE_ID in graph:
            return SYNTHESIS_NODE_ID, graph[SYNTHESIS_NODE_ID]
        return None, None
    next_id = node.next_id(answer)
    if next_id is None:
        return None, None
    return next_id, graph[next_id]


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
    choices = _guided_choices(node)
    if choices:
        extra_body["guided_choice"] = choices

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
    answer = normalize_extraction_answer(raw, node)
    if answer is not None:
        return answer

    if retry:
        retry_suffix = (
            "\n\n[Retry: respond with exactly one allowed answer key from the list. "
            "Use not_mentioned only if none apply.]"
        )
        raw = _call(retry_suffix)
        answer = normalize_extraction_answer(raw, node)
        if answer is not None:
            return answer

    raise GraphWalkError(
        f"Invalid answer for node {node.id!r}: {raw!r}; valid options: {_guided_choices(node)}"
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
        if answer == NOT_MENTIONED and node.id == SYNTHESIS_NODE_ID:
            answer = "descriptive_only"
        elif answer == NOT_MENTIONED and node.id == DIAGNOSIS_NODE_ID:
            answer = "descriptive"
        next_id, next_node = _resolve_next_node(node, answer, graph)
        _append_step(steps, node, answer, graph, next_id)

        if next_id is None or next_node is None:
            break

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
