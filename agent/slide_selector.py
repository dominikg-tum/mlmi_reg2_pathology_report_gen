"""Paper-faithful SS-LLM selection of one clinically significant slide chain."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SELECTION_SYSTEM_PROMPT = """You are a pathology case adjudicator.
Choose exactly one slide whose diagnostic chain is the best case-level prediction.

Policy, in priority order:
1. Prefer the most clinically significant supported category:
   malignant > premalignant > benign > non-neoplastic/reactive > descriptive.
2. Prefer a specific, confident diagnosis over a vague or descriptive chain.
3. Do not combine chains and do not invent findings.
4. When evidence is equivalent, choose the first slide.

Return JSON only:
{"chosen_slide_id": "<one provided slide id>", "rationale": "<one short sentence>"}
"""


@dataclass(frozen=True)
class SlideSelection:
    chosen_slide_id: str
    rationale: str
    method: str


def _chain_text(chain: dict[str, Any]) -> str:
    lines: list[str] = []
    for step in chain.get("chain-of-thought") or []:
        lines.append(
            f"{step.get('node_id', '')}: {step.get('question', '')} "
            f"-> {step.get('answer', '')}"
        )
    report = str(chain.get("report", "") or "").strip()
    if report:
        lines.append(f"Per-slide report: {report}")
    return "\n".join(lines)


def build_selection_prompt(
    chains: list[dict[str, Any]],
    slide_ids: list[str],
) -> str:
    if not chains or len(chains) != len(slide_ids):
        raise ValueError("chains and slide_ids must be non-empty and have equal length")
    sections = [
        f"## Slide {slide_id}\n{_chain_text(chain)}"
        for slide_id, chain in zip(slide_ids, chains, strict=True)
    ]
    return "Select the case-level chain from these candidates:\n\n" + "\n\n".join(
        sections
    )


def _diagnosis_rank(chain: dict[str, Any]) -> int:
    ranks = {
        "malignant": 5,
        "premalignant": 4,
        "benign": 3,
        "non_neoplastic": 2,
        "non-neoplastic": 2,
        "reactive": 2,
        "descriptive": 1,
    }
    for step in reversed(chain.get("chain-of-thought") or []):
        if step.get("node_id") == "diagnosis":
            return ranks.get(str(step.get("answer", "")).strip().lower(), 0)
    return 0


def fallback_selection(
    chains: list[dict[str, Any]],
    slide_ids: list[str],
) -> SlideSelection:
    """Deterministic fallback based on the graph's final diagnosis category."""
    if not chains or len(chains) != len(slide_ids):
        raise ValueError("chains and slide_ids must be non-empty and have equal length")
    chosen_index = max(range(len(chains)), key=lambda i: _diagnosis_rank(chains[i]))
    return SlideSelection(
        chosen_slide_id=slide_ids[chosen_index],
        rationale="Selected by diagnosis severity fallback; ties use slide order.",
        method="severity_fallback",
    )


def _parse_json_response(raw: str, slide_ids: list[str]) -> SlideSelection | None:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    chosen = str(parsed.get("chosen_slide_id", "")).strip()
    if chosen not in slide_ids:
        return None
    rationale = str(parsed.get("rationale", "")).strip() or "Selected by SS-LLM."
    return SlideSelection(chosen, rationale, "llm")


def select_slide_chain(
    chains: list[dict[str, Any]],
    slide_ids: list[str],
    *,
    backend: Any | None = None,
) -> SlideSelection:
    """Choose one slide chain. Uses the VLM/LLM backend, with a stable fallback."""
    if not chains or len(chains) != len(slide_ids):
        raise ValueError("chains and slide_ids must be non-empty and have equal length")
    if len(chains) == 1:
        return SlideSelection(
            slide_ids[0],
            "Only one physical slide is available.",
            "single_slide",
        )

    prompt = build_selection_prompt(chains, slide_ids)
    raw = ""
    try:
        if backend is not None and hasattr(backend, "client"):
            response = backend.client.chat.completions.create(
                model=backend.model,
                messages=[
                    {"role": "system", "content": SELECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            raw = (response.choices[0].message.content or "").strip()
        elif backend is not None and hasattr(backend, "_generate"):
            raw = backend._generate(SELECTION_SYSTEM_PROMPT, prompt, [])
    except Exception:
        raw = ""

    parsed = _parse_json_response(raw, slide_ids)
    return parsed or fallback_selection(chains, slide_ids)


def selection_from_case_chain(
    case_chain: dict[str, Any],
    physical_slides: list[str],
) -> SlideSelection | None:
    """Recover a stored pick so metadata loss never forces a new selection."""
    if case_chain.get("fusion") != "ss_llm_pick":
        return None
    chosen = str(case_chain.get("selected_slide_id", "")).strip()
    if chosen not in physical_slides:
        return None
    return SlideSelection(
        chosen_slide_id=chosen,
        rationale=str(case_chain.get("selection_rationale", "") or "").strip()
        or "Restored from stored case chain.",
        method=str(case_chain.get("selection_method", "") or "").strip() or "restored",
    )


def build_selected_case_chain(
    selected_chain: dict[str, Any],
    *,
    case_key: str,
    physical_slides: list[str],
    selection: SlideSelection,
) -> dict[str, Any]:
    """Copy one per-slide chain into the case eval slot without concatenation."""
    result = dict(selected_chain)
    result["slide_id"] = case_key
    result["physical_slides"] = list(physical_slides)
    result["selected_slide_id"] = selection.chosen_slide_id
    result["selection_rationale"] = selection.rationale
    result["selection_method"] = selection.method
    result["fusion"] = "ss_llm_pick"
    return result


def selection_metadata(
    *,
    case_key: str,
    physical_slides: list[str],
    selection: SlideSelection,
    skipped_slides: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "case_key": case_key,
        "physical_slides": list(physical_slides),
        "chosen_slide_id": selection.chosen_slide_id,
        "rationale": selection.rationale,
        "selection_method": selection.method,
        "fusion": "ss_llm_pick",
    }
    if skipped_slides:
        meta["skipped_slides"] = list(skipped_slides)
    return meta


def write_case_meta(
    path: Path,
    *,
    case_key: str,
    physical_slides: list[str],
    selection: SlideSelection,
    skipped_slides: list[dict[str, Any]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = selection_metadata(
        case_key=case_key,
        physical_slides=physical_slides,
        selection=selection,
        skipped_slides=skipped_slides,
    )
    path.write_text(json.dumps(meta, indent=2) + "\n")
