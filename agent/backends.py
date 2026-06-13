"""VLM backends: zero-shot Qwen, dummy, future LoRA."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from agent.types import Step

if TYPE_CHECKING:
    import openai
from graph.schema import InteractionType, Node
from vision.backends import VisualBundle
from vision.vlm_messages import build_user_content


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
        ctx = f"\n{extra_context}" if extra_context else ""
        prompt = (
            f"You are analyzing a uterine pathology slide.{visual_note}\n"
            f"{history}{ctx}\n"
            f"Current question: {node.question}\nAnswer:"
        )
        image_paths = _visual_image_paths(visual)
        content = build_user_content(prompt, image_paths)

        extra_body: dict[str, Any] = {}
        if node.interaction in (InteractionType.SINGLE_SELECT, InteractionType.BOOLEAN):
            if node.options:
                extra_body["guided_choice"] = node.options

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            logprobs=True,
            extra_body=extra_body or None,
        )
        choice = resp.choices[0]
        answer = (choice.message.content or "").strip()
        confidence = _first_token_prob(choice)
        return answer, confidence


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
        return "\n[Visual: whole-slide thumbnail attached.]"
    return ""


def _visual_image_paths(visual: VisualBundle | None) -> list:
    if visual is None:
        return []
    paths = []
    if visual.thumbnail_path:
        paths.append(visual.thumbnail_path)
    paths.extend(visual.patch_paths)
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
