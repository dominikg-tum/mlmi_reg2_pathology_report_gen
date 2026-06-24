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
    def __init__(
        self,
        client: Any,
        model: str,
        *,
        use_guided_choice: bool = True,
        request_logprobs: bool = True,
    ):
        self.client = client
        self.model = model
        self.use_guided_choice = use_guided_choice
        self.request_logprobs = request_logprobs

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
        embedding_context = _visual_embedding_context(visual)
        if embedding_context:
            prompt_parts.append(f"Embedding features:\n{embedding_context}")
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
            prompt_parts.append(
                "Combine the visual findings and all prior diagnostic answers into a "
                "concise final pathology report. State the specimen/procedure when "
                "supported, followed by the principal diagnosis and key qualifiers. "
                "Do not mention the reasoning process or answer keys."
            )
        elif node.interaction == InteractionType.FREE_TEXT:
            prompt_parts.append("Return a concise pathology answer.")
        else:
            prompt_parts.append("Return exactly one allowed answer key.")
        prompt = "\n\n".join(prompt_parts)
        image_paths = _visual_image_paths(visual)
        content = build_user_content(prompt, image_paths)

        extra_body: dict[str, Any] = {}
        if (
            self.use_guided_choice
            and node.interaction in (InteractionType.SINGLE_SELECT, InteractionType.BOOLEAN)
        ):
            if node.options:
                extra_body["guided_choice"] = node.options

        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
        }
        if self.request_logprobs:
            request["logprobs"] = True
        if extra_body:
            request["extra_body"] = extra_body

        resp = self.client.chat.completions.create(**request)
        choice = resp.choices[0]
        answer = (choice.message.content or "").strip()
        confidence = _first_token_prob(choice) if self.request_logprobs else 1.0
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
    paths.extend(visual.patch_paths)
    return paths


def _visual_embedding_context(visual: VisualBundle | None) -> str:
    if visual is None:
        return ""
    return str(visual.metadata.get("embedding_context", "")).strip()


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
