"""VLM backends: zero-shot Qwen, dummy, future LoRA."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from agent.types import Step

if TYPE_CHECKING:
    import openai
from graph.schema import InteractionType, Node
from vision.backends import VisualBundle


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
        visual_note = ""
        if visual and visual.thumbnail_path:
            visual_note = f"\n[Slide thumbnail: {visual.thumbnail_path}]"
        ctx = f"\n{extra_context}" if extra_context else ""
        prompt = (
            f"You are analyzing a uterine pathology slide.{visual_note}\n"
            f"{history}{ctx}\n"
            f"Current question: {node.question}\nAnswer:"
        )
        extra_body: dict[str, Any] = {}
        if node.interaction in (InteractionType.SINGLE_SELECT, InteractionType.BOOLEAN):
            if node.options:
                extra_body["guided_choice"] = node.options

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            logprobs=True,
            extra_body=extra_body or None,
        )
        choice = resp.choices[0]
        answer = (choice.message.content or "").strip()
        confidence = _first_token_prob(choice)
        return answer, confidence


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
