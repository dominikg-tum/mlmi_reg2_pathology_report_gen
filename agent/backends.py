"""VLM backends: zero-shot Qwen, dummy, future LoRA."""

from __future__ import annotations

from pathlib import Path
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


def build_answer_prompt(
    node: Node,
    history: str,
    visual_note: str,
    extra_context: str = "",
) -> str:
    """Assemble the user-turn text for a graph node.

    Single source of truth shared by ``ZeroShotQwenBackend`` (vLLM API), the local
    ``FineTunedBackend``, and the LoRA training-data prompt builder so train and serve
    see byte-identical prompts. ``history`` is the prior-answer Q/A block, ``visual_note``
    describes the attached images (see :func:`_visual_prompt_note` /
    :func:`visual_note_for_paths`).
    """
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
    return "\n\n".join(prompt_parts)


def system_prompt_for(node: Node) -> str:
    return REPORT_SYSTEM_PROMPT if node.node_kind == NodeKind.REPORT else SYSTEM_PROMPT


def memory_history(memory: list[Step]) -> str:
    return "\n".join(f"Q: {s.question}\nA: {s.answer}" for s in memory)


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
        history = memory_history(memory)
        visual_note = _visual_prompt_note(visual)
        prompt = build_answer_prompt(node, history, visual_note, extra_context)
        image_paths = _visual_image_paths(visual)
        content = build_user_content(prompt, image_paths)

        extra_body: dict[str, Any] = {}
        if node.interaction in (InteractionType.SINGLE_SELECT, InteractionType.BOOLEAN):
            if node.options:
                extra_body["guided_choice"] = node.options

        system_prompt = system_prompt_for(node)
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


def visual_note_for_paths(image_paths: list) -> str:
    """Reconstruct the 'Visual evidence' note from already-materialized image paths.

    Used by the LoRA training-data prompt builder, which no longer has the live
    ``VisualBundle`` — only the saved image files. Mirrors the ``thumbnail`` branch of
    :func:`_visual_prompt_note`: the first image is the whole-slide thumbnail (providers
    always place it first in :func:`_visual_image_paths`) and the rest are patches.
    """
    n = len(image_paths)
    if n == 0:
        return ""
    patch_count = n - 1
    suffix = f" and {patch_count} retrieved patch images" if patch_count else ""
    return f" whole-slide thumbnail{suffix} attached."


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


class FineTunedBackend:
    """Local HF inference for a LoRA-fine-tuned Qwen3-VL adapter (train/serve parity).

    Loads the base VLM + PEFT adapter once, then answers each graph node with the SAME
    system prompt, visual-evidence note, episodic history, and question layout used both
    at zero-shot serving (``ZeroShotQwenBackend``) and during LoRA data generation. Heavy
    deps (torch/transformers/peft) are imported lazily so the module still imports on a
    laptop with no GPU.
    """

    def __init__(
        self,
        adapter_dir: str,
        *,
        base_model: str,
        max_new_tokens: int = 512,
        device: str | None = None,
        dtype: str = "bfloat16",
    ):
        self.adapter_dir = str(adapter_dir)
        self.base_model = str(base_model)
        self.max_new_tokens = max_new_tokens
        self._device = device
        self._dtype = dtype
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as _AutoVLM
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as _AutoVLM

        torch_dtype = getattr(torch, self._dtype, torch.bfloat16)
        device_map = "auto" if self._device is None else None

        base = _AutoVLM.from_pretrained(
            self.base_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, self.adapter_dir)
        model.eval()
        if device_map is None:
            model.to(self._device)
        self._model = model
        self._processor = AutoProcessor.from_pretrained(
            self.base_model, trust_remote_code=True
        )

    def answer(
        self,
        node: Node,
        visual: VisualBundle | None,
        memory: list[Step],
        *,
        extra_context: str = "",
    ) -> tuple[str, float]:
        self._ensure_loaded()
        import torch

        from vision.vlm_messages import _load_rgb_image

        history = memory_history(memory)
        visual_note = _visual_prompt_note(visual)
        prompt = build_answer_prompt(node, history, visual_note, extra_context)
        image_paths = [p for p in _visual_image_paths(visual) if p is not None]

        user_content: list[dict[str, Any]] = [
            {"type": "image"} for _ in image_paths
        ]
        user_content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt_for(node)}]},
            {"role": "user", "content": user_content},
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = [_load_rgb_image(Path(p)) for p in image_paths] or None
        inputs = self._processor(
            text=[text], images=images, return_tensors="pt", padding=True
        ).to(self._model.device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return (decoded[0].strip() if decoded else ""), 1.0
