"""Deterministic graph traversal with visual + memory hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.backends import AnswerBackend
from agent.correction import check_consistency, should_retry
from agent.memory import CaseMemory, GraphStore, JsonGraphStore
from agent.types import Step
from graph import GRAPH, ROOT_ID, Node
from graph.schema import InteractionType
from retrieval.base import PatchRetriever, get_retriever
from vision.backends import VisualBundle
from vision.cache import SlideCache
from vision.navigation import get_navigator


def build_query(node: Node, memory: list[Step]) -> str:
    if not memory:
        return node.question
    context = "; ".join(f"{s.question} -> {s.answer}" for s in memory)
    return f"Given: {context}. {node.question}"


def _format_answer(node: Node, raw: str) -> str:
    if node.interaction == InteractionType.MULTI_SELECT:
        return raw.strip()
    return raw.strip()


def traverse(
    backend: AnswerBackend,
    *,
    graph_store: GraphStore | None = None,
    graph: dict[str, Node] | None = None,
    root_id: str | None = None,
    case_memory: CaseMemory | None = None,
    slide_cache: SlideCache | None = None,
    visual_method: str = "thumbnail",
    retriever_method: str = "none",
    navigator_method: str = "graph_guided",
    cache_root: Path | None = None,
    max_steps: int = 64,
    confidence_threshold: float = 0.65,
) -> list[Step]:
    graph = graph or GRAPH
    root_id = root_id or ROOT_ID
    store = graph_store or JsonGraphStore(graph)
    mem = case_memory or CaseMemory()
    retriever: PatchRetriever | None = get_retriever(retriever_method)
    navigator = get_navigator(
        navigator_method,
        visual=visual_method,
        cache_root=cache_root,
    )

    node = store.get_node(root_id)
    steps: list[Step] = []

    for _ in range(max_steps):
        query = build_query(node, steps)
        visual_bundle: VisualBundle = navigator.select_visual_bundle(
            node,
            slide_cache,
            steps,
            query=query,
            retriever=retriever if node.needs_patch_retrieval() else None,
        )
        extra = mem.retrieve_context(node, query)
        answer, confidence = backend.answer(
            node, visual_bundle, steps, extra_context=extra
        )
        answer = _format_answer(node, answer)

        if should_retry(confidence, confidence_threshold):
            answer_retry, conf_retry = backend.answer(
                node, visual_bundle, steps, extra_context=extra + "\n[Retry: reconsider]"
            )
            if conf_retry > confidence:
                answer, confidence = _format_answer(node, answer_retry), conf_retry

        _ = check_consistency(steps, node, answer)

        next_id = store.next(node.id, answer)
        next_q = ""
        if next_id is not None:
            next_q = store.get_node(next_id).question

        steps.append(
            Step(node.id, node.question, answer, confidence, next_question=next_q)
        )
        mem.append(node.id, node.question, answer)

        if next_id is None:
            break
        node = store.get_node(next_id)
    else:
        raise RuntimeError(f"traversal exceeded {max_steps} steps (cycle?)")

    return steps


def chain_to_dict(steps: list[Step], slide_id: str = "") -> dict[str, Any]:
    return {
        "slide_id": slide_id,
        "chain-of-thought": [
            {
                "question": s.question,
                "answer": s.answer,
                "next_question": s.next_question,
            }
            for s in steps
        ],
        "report": steps[-1].answer if steps else "",
    }
