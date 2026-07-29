"""Naive baseline: thumbnail → one-shot CoT (no graph), then SS-LLM Pick."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.report_writer import write_case_chain
from agent.slide_selector import (
    build_selected_case_chain,
    select_slide_chain,
    selection_from_case_chain,
    write_case_meta,
)
from baselines.agent_runner import (
    build_backend,
    build_selector_backend,
    load_paths_config,
    load_vision_cache_root,
)
from extraction.case_ids import (
    CaseSpec,
    case_run_dir,
    case_spec_from_key,
    physical_run_dir,
)
from graph.schema import InteractionType, Node, NodeKind, Tier, VisualPolicy
from vision.backends import VisualBundle, get_visual_provider
from vision.cache import build_slide_cache
from vision.thumbnail import _resolve_wsi_path

NAIVE_SYSTEM = (
    "You are a board-certified pathologist. Given only a whole-slide thumbnail of a "
    "uterine specimen, produce a short chain-of-thought Q/A then a final impression.\n"
    "Use this format exactly:\n"
    "Q1: What tissue/compartment is visible?\n"
    "A1: <answer>\n"
    "Q2: What are the main pathologic findings?\n"
    "A2: <answer>\n"
    "FINAL: <one-paragraph impression>\n\n"
    "Example:\n"
    "Q1: What tissue/compartment is visible?\n"
    "A1: Endometrial curettage fragments with stroma and glands.\n"
    "Q2: What are the main pathologic findings?\n"
    "A2: Proliferative endometrium without atypia or malignancy.\n"
    "FINAL: Proliferative endometrium; no evidence of hyperplasia or carcinoma.\n"
)

NAIVE_USER = (
    "Analyze the attached whole-slide thumbnail. Output Q1/A1, Q2/A2, and FINAL only."
)


def _naive_node() -> Node:
    return Node(
        id="naive_oneshot",
        label="Naive thumbnail CoT",
        question=NAIVE_USER,
        tier=Tier.GLOBAL_FEATURES,
        node_kind=NodeKind.GLOBAL,
        interaction=InteractionType.FREE_TEXT,
        description="One-shot thumbnail baseline without graph traversal.",
        visual_policy=VisualPolicy.THUMBNAIL_ONLY,
        requires_visual_evidence=True,
        is_leaf=True,
        root=True,
    )


def _parse_naive_response(text: str, slide_id: str) -> dict[str, Any]:
    """Best-effort parse of Q/A blocks; fall back to a single free-text step."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    qa: list[tuple[str, str]] = []
    final = ""
    pending_q = ""
    for ln in lines:
        upper = ln.upper()
        if upper.startswith("FINAL:"):
            final = ln.split(":", 1)[-1].strip()
            continue
        if upper.startswith("Q") and ":" in ln:
            pending_q = ln.split(":", 1)[-1].strip()
            continue
        if upper.startswith("A") and ":" in ln and pending_q:
            qa.append((pending_q, ln.split(":", 1)[-1].strip()))
            pending_q = ""
            continue

    steps: list[dict[str, Any]] = []
    if qa:
        for i, (q, a) in enumerate(qa):
            steps.append(
                {
                    "node_id": f"naive_q{i + 1}",
                    "question": q,
                    "answer": a,
                    "next_question": "",
                }
            )
    else:
        steps.append(
            {
                "node_id": "naive_oneshot",
                "question": "Thumbnail-only diagnosis (no graph)",
                "answer": text.strip(),
                "next_question": "",
            }
        )
    if final:
        steps.append(
            {
                "node_id": "naive_final",
                "question": "Final impression",
                "answer": final,
                "next_question": "",
            }
        )

    return {
        "slide_id": slide_id,
        "chain-of-thought": steps,
        "node_path": [s["node_id"] for s in steps],
        "report": final,
        "baseline": "naive",
    }


def run_naive_oneshot(
    slide_id: str,
    *,
    backend: str = "qwen",
    cache_root: Path | None = None,
    wsi_data_dir: Path | None = None,
) -> dict[str, Any]:
    """Single-slide thumbnail → CoT chain dict (no graph)."""
    cfg = load_paths_config()
    wsi_data_dir = wsi_data_dir or Path(cfg["cluster"]["data_dir"])
    cache_root = cache_root or load_vision_cache_root()
    answer_backend = build_backend(backend, cfg)

    slide_cache = (
        build_slide_cache(cache_root, slide_id) if slide_id and cache_root else None
    )
    wsi_path = _resolve_wsi_path(
        slide_cache, wsi_path=None, wsi_data_dir=wsi_data_dir
    )
    provider = get_visual_provider(
        "thumbnail",
        cache_root,
        wsi_path=wsi_path,
        wsi_data_dir=wsi_data_dir,
    )
    node = _naive_node()
    visual: VisualBundle = provider.for_node(
        node, slide_cache, query=node.question, retriever=None
    )

    if backend == "dummy":
        text = (
            "Q1: What tissue/compartment is visible?\n"
            "A1: Uterine tissue (dummy).\n"
            "Q2: What are the main pathologic findings?\n"
            "A2: No specific findings (dummy).\n"
            "FINAL: Dummy naive thumbnail impression."
        )
    else:
        image_paths = []
        if visual.thumbnail_path:
            image_paths.append(visual.thumbnail_path)

        # Always use the naive CoT system prompt (qwen client or FineTunedBackend).
        if hasattr(answer_backend, "client"):
            from vision.vlm_messages import build_user_content

            content = build_user_content(NAIVE_USER, image_paths)
            resp = answer_backend.client.chat.completions.create(
                model=answer_backend.model,
                messages=[
                    {"role": "system", "content": NAIVE_SYSTEM},
                    {"role": "user", "content": content},
                ],
                temperature=0.0,
            )
            text = (resp.choices[0].message.content or "").strip()
        elif hasattr(answer_backend, "_generate"):
            text = answer_backend._generate(NAIVE_SYSTEM, NAIVE_USER, image_paths)
        else:
            # Last resort: put format instructions in the node question.
            node.question = f"{NAIVE_SYSTEM}\n\n{NAIVE_USER}"
            text, _ = answer_backend.answer(node, visual, [])

    return _parse_naive_response(text, slide_id)


def run_naive_case(
    case: CaseSpec | str,
    *,
    runs_dir: Path,
    backend: str = "qwen",
    skip_existing: bool = False,
    cache_root: Path | None = None,
    wsi_data_dir: Path | None = None,
) -> Path:
    """SS-LLM naive: one-shot per physical slide, then pick one case chain."""
    if isinstance(case, str):
        case = case_spec_from_key(case)

    case_dir = case_run_dir(runs_dir, case.case_key)
    case_chain_path = case_dir / "cot_chain.json"

    def _all_physical_chains_exist() -> bool:
        return all(
            (physical_run_dir(runs_dir, case.case_key, pid) / "cot_chain.json").exists()
            for pid in case.physical_slides
        )

    if skip_existing and case_chain_path.exists() and _all_physical_chains_exist():
        stored = selection_from_case_chain(
            json.loads(case_chain_path.read_text()), case.physical_slides
        )
        if stored is not None:
            meta_path = case_dir / "case_meta.json"
            if not meta_path.exists():
                write_case_meta(
                    meta_path,
                    case_key=case.case_key,
                    physical_slides=case.physical_slides,
                    selection=stored,
                )
            return case_chain_path

    chains: list[dict[str, Any]] = []
    for physical_id in case.physical_slides:
        phys_dir = physical_run_dir(runs_dir, case.case_key, physical_id)
        phys_chain = phys_dir / "cot_chain.json"
        if skip_existing and phys_chain.exists():
            chains.append(json.loads(phys_chain.read_text()))
            continue

        chain = run_naive_oneshot(
            physical_id,
            backend=backend,
            cache_root=cache_root,
            wsi_data_dir=wsi_data_dir,
        )
        phys_dir.mkdir(parents=True, exist_ok=True)
        phys_chain.write_text(json.dumps(chain, indent=2) + "\n")
        report = str(chain.get("report", "") or "").strip()
        if report:
            (phys_dir / "report.txt").write_text(report + "\n")
        chains.append(chain)

    selection = select_slide_chain(
        chains,
        case.physical_slides,
        backend=build_selector_backend(backend),
    )
    selected_index = case.physical_slides.index(selection.chosen_slide_id)
    case_chain = build_selected_case_chain(
        chains[selected_index],
        case_key=case.case_key,
        physical_slides=case.physical_slides,
        selection=selection,
    )
    write_case_chain(case_chain_path, case_chain)
    write_case_meta(
        case_dir / "case_meta.json",
        case_key=case.case_key,
        physical_slides=case.physical_slides,
        selection=selection,
    )
    return case_chain_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Naive thumbnail one-shot CoT (no graph), SS-LLM case-aware."
    )
    parser.add_argument(
        "--slide-id",
        required=True,
        help="Case key (GT slide_id; may be comma-separated multi-WSI)",
    )
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--backend", choices=["dummy", "qwen", "finetuned"], default="qwen")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    from baselines.agent_runner import default_runs_dir

    runs_dir = args.runs_dir or default_runs_dir()
    path = run_naive_case(
        args.slide_id,
        runs_dir=runs_dir,
        backend=args.backend,
        skip_existing=args.skip_existing,
    )
    print(f"Naive baseline complete: {path}")


if __name__ == "__main__":
    main()
