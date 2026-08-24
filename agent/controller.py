"""Deterministic graph traversal with visual + memory hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agent.answers import normalize_answer
from agent.backends import AnswerBackend
from agent.correction import check_consistency, should_retry
from agent.memory import CaseMemory, GraphStore, JsonGraphStore
from agent.types import Step
from graph import GRAPH, ROOT_ID, Node
from graph.schema import InteractionType, NodeKind
from retrieval.base import PatchRetriever, get_retriever
from vision.backends import VisualBundle
from vision.cache import SlideCache
from vision.mag_config import fixed_retrieval_pool
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
    search_all_patches: bool | None = None,
    retrieval_log: list[dict[str, Any]] | None = None,
    fixed_visual_bundle: VisualBundle | None = None,
    node_react: bool = False,
    structured_answer: bool = False,
    paired_regions: bool = False,
    react_max_iters: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[Step]:
    graph = graph or GRAPH
    root_id = root_id or ROOT_ID
    store = graph_store or JsonGraphStore(graph)
    mem = case_memory or CaseMemory()
    retriever: PatchRetriever | None = None
    navigator = None
    if fixed_visual_bundle is None:
        retriever_kwargs: dict[str, Any] = {}
        if retriever_method in ("titan_cosine", "graph_guided"):
            from vision.encoders.titan import TitanEncoder

            _encoder = TitanEncoder()
            retriever_kwargs["text_encoder"] = _encoder.encode_text
            retriever_kwargs["search_all_patches"] = search_all_patches
        retriever = get_retriever(retriever_method, **retriever_kwargs)
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
        choice_node = node.interaction in (
            InteractionType.SINGLE_SELECT,
            InteractionType.BOOLEAN,
        )
        # run_node_react does its own retrieval per iteration, so the navigator
        # must not retrieve (and pay for patch image IO) on these nodes.
        react_skip_reasons: list[str] = []
        if node_react:
            if not choice_node:
                react_skip_reasons.append("not_choice_node")
            if not node.needs_patch_retrieval():
                react_skip_reasons.append("no_patch_retrieval")
            if not hasattr(backend, "complete_json"):
                react_skip_reasons.append("backend_no_complete_json")
            if retriever is None:
                react_skip_reasons.append("no_retriever")
            if slide_cache is None:
                react_skip_reasons.append("no_slide_cache")
            if fixed_visual_bundle is not None:
                react_skip_reasons.append("fixed_visual_bundle")
        use_react = bool(node_react) and not react_skip_reasons
        structured_skip_reasons: list[str] = []
        if structured_answer and not use_react:
            if not choice_node:
                structured_skip_reasons.append("not_choice_node")
            if not hasattr(backend, "complete_json"):
                structured_skip_reasons.append("backend_no_complete_json")
        use_structured = bool(structured_answer) and not use_react and not structured_skip_reasons
        answer_branch = "plain"
        answer_branch_skip_reason = ""
        if node_react and react_skip_reasons:
            answer_branch_skip_reason = ",".join(react_skip_reasons)
        elif structured_answer and structured_skip_reasons:
            answer_branch_skip_reason = ",".join(structured_skip_reasons)
        if fixed_visual_bundle is not None:
            visual_bundle = fixed_visual_bundle
        else:
            visual_bundle = navigator.select_visual_bundle(
                node,
                slide_cache,
                steps,
                query=query,
                retriever=(
                    retriever if node.needs_patch_retrieval() and not use_react else None
                ),
            )
        if retrieval_log is not None:
            patches_meta = visual_bundle.metadata.get("retrieved_patches")
            if patches_meta:
                retrieval_log.append(
                    {
                        "node_id": node.id,
                        "pool": fixed_retrieval_pool(),
                        "node_zoom_hint": node.mag_band,
                        "query": node.retrieval_text,
                        "patches": patches_meta,
                    }
                )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "node_started",
                    "node_id": node.id,
                    "question": node.question,
                    "steps": list(steps),
                }
            )
        extra = mem.retrieve_context(node, query)
        semantic_extra = mem.semantic_context(node, query)
        answer = ""
        confidence = 0.0
        last_raw = ""
        node_traces: list[dict[str, Any]] = []
        # Every ReAct attempt retrieves, so the log keeps all of them even when
        # only the last attempt's traces end up on the chain step.
        react_log: list[tuple[int, dict[str, Any]]] = []
        react_failed = False
        for attempt in range(max_answer_attempts):
            retry_note = ""
            if attempt:
                retry_note = (
                    "\nRetry instruction: your previous response was not a valid graph "
                    "answer. Return exactly one allowed answer key and nothing else."
                )

            if use_react and not react_failed:
                from agent.node_react import run_node_react

                react = run_node_react(
                    node,
                    backend=backend,
                    retriever=retriever,
                    slide_cache=slide_cache,
                    wsi_path=wsi_path,
                    prior_steps=steps,
                    paired_regions=paired_regions,
                    extra_context=semantic_extra,
                    max_iters=react_max_iters,
                )
                last_raw = react.answer_key
                node_traces = react.node_traces
                react_log.extend((attempt, tr) for tr in react.node_traces)
                normalized = normalize_answer(react.answer_key, node)
                if normalized is not None:
                    answer, confidence = normalized, float(react.confidence)
                    answer_branch = "react"
                    break
                # Retry on the cheap single-call path instead of replaying the
                # whole A/B/C loop; keep the evidence ReAct already retrieved.
                react_failed = True
                answer_branch_skip_reason = "react_invalid_answer"
                if react.bundle is not None:
                    visual_bundle = react.bundle
                continue

            if use_structured:
                from agent import prompts

                user_prompt = prompts.format_step_a_user(
                    node=node,
                    prior_steps=[(s.node_id, s.answer) for s in steps],
                    extra_context=semantic_extra,
                )
                draft, raw_confidence, _raw = backend.complete_json(
                    node,
                    visual_bundle,
                    system_prompt=prompts.STEP_A_SYSTEM,
                    user_prompt=user_prompt,
                )
                answer_key = str(draft.get("answer_key", "")).strip()
                normalized = normalize_answer(answer_key, node)
                if normalized is None:
                    continue
                answer = normalized
                confidence = float(draft.get("confidence", raw_confidence) or raw_confidence)
                last_raw = answer_key
                answer_branch = "structured"
                break

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
                answer_branch = "plain"
            if not should_retry(confidence, confidence_threshold):
                break

        if retrieval_log is not None:
            for react_attempt, react_trace in react_log:
                if not react_trace.get("patches"):
                    continue
                retrieval_log.append(
                    {
                        "node_id": node.id,
                        "pool": react_trace.get("pool", fixed_retrieval_pool()),
                        "node_zoom_hint": node.mag_band,
                        "query": react_trace.get("query", node.retrieval_text),
                        "answer_attempt": react_attempt,
                        "react_iter": react_trace.get("iter"),
                        "patches": react_trace["patches"],
                    }
                )

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

        step = Step(
            node.id,
            node.question,
            answer,
            confidence,
            next_question=next_q,
            raw_answer=last_raw,
            node_traces=node_traces,
            answer_branch=answer_branch,
            answer_branch_skip_reason=answer_branch_skip_reason,
        )
        steps.append(step)
        mem.append(node.id, node.question, answer)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "node_finished",
                    "node_id": node.id,
                    "question": node.question,
                    "answer": answer,
                    "confidence": confidence,
                    "raw_answer": last_raw,
                    "next_node_id": next_id,
                    "next_question": next_q,
                    "steps": list(steps),
                }
            )

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
                "confidence": s.confidence,
                "raw_answer": s.raw_answer,
                "next_question": s.next_question,
                "node_traces": s.node_traces,
                "answer_branch": s.answer_branch,
                "answer_branch_skip_reason": s.answer_branch_skip_reason,
            }
            for s in steps
        ],
        "node_path": [s.node_id for s in steps],
        "report": report,
    }
