"""Deterministic graph traversal with visual + memory hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.answers import normalize_answer
from agent.backends import AnswerBackend
from agent.correction import check_consistency, should_retry
from agent.memory import CaseMemory, GraphStore, JsonGraphStore
from agent.types import Step
from graph import GRAPH, ROOT_ID, Node
from graph.schema import NodeKind
from retrieval.base import PatchRetriever, get_retriever
from vision.backends import VisualBundle
from vision.cache import SlideCache
from vision.navigation import get_navigator


def build_query(node: Node, memory: list[Step]) -> str:
    if not memory:
        return node.question
    context = "; ".join(f"{s.question} -> {s.answer}" for s in memory)
    return f"Given: {context}. {node.question}"


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
    wsi_path: Path | None = None,
    wsi_data_dir: Path | None = None,
    max_steps: int = 64,
    confidence_threshold: float = 0.65,
    max_answer_attempts: int = 2,
    skip_report_nodes: bool = False,
    search_all_patches: bool = False,
    retrieval_log: list[dict[str, Any]] | None = None,
) -> list[Step]:
    graph = graph or GRAPH
    root_id = root_id or ROOT_ID
    store = graph_store or JsonGraphStore(graph)
    mem = case_memory or CaseMemory()
    retriever_kwargs: dict[str, Any] = {}
    if retriever_method in ("titan_cosine", "graph_guided"):
        from vision.encoders.titan import TitanEncoder

        _encoder = TitanEncoder()
        retriever_kwargs["text_encoder"] = _encoder.encode_text
        retriever_kwargs["search_all_patches"] = search_all_patches
    retriever: PatchRetriever | None = get_retriever(retriever_method, **retriever_kwargs)
    navigator = get_navigator(
        navigator_method,
        visual=visual_method,
        cache_root=cache_root,
        wsi_path=wsi_path,
        wsi_data_dir=wsi_data_dir,
    )

    node = store.get_node(root_id)
    steps: list[Step] = []

    for _ in range(max_steps):
        if skip_report_nodes and node.node_kind == NodeKind.REPORT:
            break

        query = build_query(node, steps)
        visual_bundle: VisualBundle = navigator.select_visual_bundle(
            node,
            slide_cache,
            steps,
            query=query,
            retriever=retriever if node.needs_patch_retrieval() else None,
        )
        if retrieval_log is not None:
            patches_meta = visual_bundle.metadata.get("retrieved_patches")
            if patches_meta:
                retrieval_log.append(
                    {
                        "node_id": node.id,
                        "zoom_level": node.mag_band,
                        "query": node.retrieval_text,
                        "patches": patches_meta,
                    }
                )
        extra = mem.retrieve_context(node, query)
        answer = ""
        confidence = 0.0
        last_raw = ""
        for attempt in range(max_answer_attempts):
            retry_note = ""
            if attempt:
                retry_note = (
                    "\nRetry instruction: your previous response was not a valid graph "
                    "answer. Return exactly one allowed answer key and nothing else."
                )
            raw, raw_confidence = backend.answer(
                node,
                visual_bundle,
                steps,
                extra_context=extra + retry_note,
            )
            last_raw = raw
            normalized = normalize_answer(raw, node)
            if normalized is None:
                continue
            if not answer or raw_confidence > confidence:
                answer, confidence = normalized, raw_confidence
            if not should_retry(confidence, confidence_threshold):
                break

        if not answer:
            raise ValueError(
                f"VLM failed to answer node {node.id!r} after "
                f"{max_answer_attempts} attempts. Last response: {last_raw!r}; "
                f"expected one of: {node.options}"
            )

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
        next_node = store.get_node(next_id)
        if skip_report_nodes and next_node.node_kind == NodeKind.REPORT:
            break
        node = next_node
    else:
        raise RuntimeError(f"traversal exceeded {max_steps} steps (cycle?)")

    return steps


def chain_to_dict(
    steps: list[Step],
    slide_id: str = "",
    *,
    include_report: bool = True,
) -> dict[str, Any]:
    report = ""
    if include_report and steps:
        report = steps[-1].answer
    return {
        "slide_id": slide_id,
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
    }
