"""VLM backends: zero-shot Qwen, dummy, future LoRA."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from agent.types import Step

if TYPE_CHECKING:
    import openai
from graph.schema import InteractionType, Node, NodeKind
from vision.backends import VisualBundle
from vision.vlm_messages import build_user_content

SYSTEM_PROMPT = (
    "You are a pathology visual question-answering assistant. Analyze only the "
    "provided uterine whole-slide overview and retrieved tissue patches. Use prior "
    "answers as context, but do not invent findings that are not supported by the "
    "images. For a multiple-choice question, return exactly one allowed answer key "
    "and no explanation."
)

REPORT_SYSTEM_PROMPT = (
    "You are a board-certified gynecologic pathologist drafting a final uterine "
    "pathology report from visual evidence and prior diagnostic answers. Write in "
    "formal narrative prose similar to a clinical pathology report: macroscopic and "
    "microscopic findings in complete sentences, then a clear diagnostic impression. "
    "Target roughly 700–800 characters when the case allows; shorter is acceptable "
    "for simple or largely unremarkable specimens. Do not use pipe-separated fields, "
    "bullet templates, or answer keys. Do not mention the reasoning graph."
)

REPORT_USER_INSTRUCTION = (
    "Draft the final pathology report as continuous narrative prose (not a template). "
    "Include, when supported by the evidence: specimen/procedure, macroscopic "
    "description, microscopic findings by compartment, and a diagnostic impression "
    "with key qualifiers (e.g. benign vs malignant, histologic type, grade, phase). "
    "Aim for about 700–800 characters for typical cases; be shorter if findings are "
    "minimal. Use professional pathology language only."
)


class AnswerBackend(Protocol):
    def answer(
        self,
        node: Node,
        visual: VisualBundle | None,
        memory: list[Step],
        *,
        extra_context: str = "",
    ) -> tuple[str, float]: ...


class ZeroShotQwenBackend:
    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def answer(
        self,
        node: Node,
        visual: VisualBundle | None,
        memory: list[Step],
        *,
        extra_context: str = "",
    ) -> tuple[str, float]:
        history = "\n".join(f"Q: {s.question}\nA: {s.answer}" for s in memory)
        visual_note = _visual_prompt_note(visual)
        prompt_parts = [f"Visual evidence:{visual_note or ' none attached.'}"]
        if history:
            prompt_parts.append(f"Prior diagnostic answers:\n{history}")
        if extra_context:
            prompt_parts.append(f"Additional context:\n{extra_context}")
        if node.description:
            prompt_parts.append(f"Diagnostic guidance:\n{node.description}")
        prompt_parts.append(f"Current question:\n{node.question}")
        if node.options:
            prompt_parts.append(
                "Allowed answer keys:\n" + "\n".join(f"- {option}" for option in node.options)
            )
        if node.node_kind == NodeKind.REPORT:
            prompt_parts.append(REPORT_USER_INSTRUCTION)
        elif node.interaction == InteractionType.FREE_TEXT:
            prompt_parts.append("Return a concise pathology answer.")
        else:
            prompt_parts.append("Return exactly one allowed answer key.")
        prompt = "\n\n".join(prompt_parts)
        image_paths = _visual_image_paths(visual)
        content = build_user_content(prompt, image_paths)

        extra_body: dict[str, Any] = {}
        if node.interaction in (InteractionType.SINGLE_SELECT, InteractionType.BOOLEAN):
            if node.options:
                extra_body["guided_choice"] = node.options

        system_prompt = REPORT_SYSTEM_PROMPT if node.node_kind == NodeKind.REPORT else SYSTEM_PROMPT
        resp = _chat_with_retry(
            self.client,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            logprobs=True,
            extra_body=extra_body or None,
        )
        choice = resp.choices[0]
        answer = (choice.message.content or "").strip()
        confidence = _first_token_prob(choice)
        return answer, confidence


def _chat_with_retry(client, **kwargs):
    import time

    last_err = None
    for attempt in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_err = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise last_err


def _visual_prompt_note(visual: VisualBundle | None) -> str:
    if visual is None:
        return ""
    mode = visual.metadata.get("visual", "")
    if mode == "slide_embed" and visual.slide_embedding is not None:
        dim = visual.metadata.get("slide_embedding_dim", "?")
        n_ev = visual.metadata.get("evidence_patch_count", 0)
        return (
            f"\n[Visual: TITAN slide embedding (dim={dim}) is precomputed offline; "
            f"attached are overview thumbnail + {n_ev} tissue evidence patches.]"
        )
    if visual.thumbnail_path:
        patch_count = len(visual.patch_paths)
        suffix = f" and {patch_count} retrieved patch images" if patch_count else ""
        return f" whole-slide thumbnail{suffix} attached."
    if visual.patch_paths:
        return f" {len(visual.patch_paths)} retrieved patch images attached."
    return " no image evidence attached."


def _visual_image_paths(visual: VisualBundle | None) -> list:
    if visual is None:
        return []
    paths = []
    if visual.thumbnail_path:
        paths.append(visual.thumbnail_path)
    patch_limit = 2 if visual.thumbnail_path else 5
    paths.extend(visual.patch_paths[:patch_limit])
    return paths


def _first_token_prob(choice) -> float:
    try:
        return float(2.718281828 ** choice.logprobs.content[0].logprob)
    except (AttributeError, IndexError, TypeError):
        return 1.0


class DummyBackend:
    def answer(
        self,
        node: Node,
        visual: VisualBundle | None,
        memory: list[Step],
        *,
        extra_context: str = "",
    ) -> tuple[str, float]:
        if node.options:
            return node.options[0], 1.0
        if node.edges:
            return next(iter(node.edges)), 1.0
        return "Sample pathology report.", 1.0
